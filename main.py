from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tqdm
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy

from config import argparser
from environment import *
from agent import UvfAgent
#from search_policy import SearchPolicy

tf.enable_v2_behavior()
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

def sac_run(config):
	import sac.core as core
	from sac.sac import sac
	from sac.utils.run_utils import setup_logger_kwargs
	from sac.utils.test_policy import load_policy, run_policy

	path = os.path.join(config.save_dir, config.exp_name)
	logger_kwargs = setup_logger_kwargs(config.exp_name, config.seed, config.save_dir)
	
	if(not os.path.isdir(path)):
		sac(lambda : gym.make(config.env), actor_critic=core.mlp_actor_critic,
	 		ac_kwargs=dict(hidden_sizes=[config.hid] * config.l),
	 		gamma=config.gamma, seed=config.seed, epochs=config.epochs,
	 		logger_kwargs=logger_kwargs)

	if(config.test_sac):
		env, get_action = load_policy(path, 
	                                  config.test_itr if config.test_itr >= 0 else 'last',
	                                  config.test_deterministic)
		run_policy(env, get_action, config.test_len, config.test_episodes, config.test_render)

def test_env(config):
	tf_env = env_load_fn(config.env_name, config.max_episode_steps, terminate_on_timeout=True)
	eval_tf_env = env_load_fn(config.env_name, config.max_episode_steps, terminate_on_timeout=True)

	time_step = tf_env.reset()
	step, num_steps, done = 0, 1000, False
	random_p = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
	while not time_step.is_last():
		action_step = random_p.action(time_step)
		time_step = tf_env.step(action_step.action)
		print("Observation : {}\n Desired Goal : {}\n Achieved Goal : {}\n".format(time_step.observation['observation'], 
			time_step.observation['desired_goal'], time_step.observation['achieved_goal']))
		step += 1
	print(step)	

# ---------- SORB Manipulation ----------

def train_eval(tf_agent, tf_env, eval_tf_env, config):
	"""A simple train and eval for UVF.  """
	
	tf.logging.info('random_seed = %d' % config.random_seed)
	np.random.seed(config.random_seed)
	random.seed(config.random_seed)
	tf.set_random_seed(config.random_seed)
	
	max_episode_steps = tf_env.pyenv.envs[0]._duration
	global_step = tf.train.get_or_create_global_step()
	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			tf_agent.collect_data_spec,
			batch_size=tf_env.batch_size)

	eval_metrics = [
		tf_metrics.AverageReturnMetric(buffer_size=config.num_eval_episodes),
	]
	
	eval_policy = tf_agent.policy
	collect_policy = tf_agent.collect_policy
	initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
			tf_env,
			collect_policy,
			observers=[replay_buffer.add_batch],
			num_steps=config.initial_collect_steps)
	
	collect_driver = dynamic_step_driver.DynamicStepDriver(
			tf_env,
			collect_policy,
			observers=[replay_buffer.add_batch],
			num_steps=1)
	
	initial_collect_driver.run = common.function(initial_collect_driver.run)
	collect_driver.run = common.function(collect_driver.run)
	tf_agent.train = common.function(tf_agent.train)
	
	initial_collect_driver.run()
	
	time_step = None
	policy_state = collect_policy.get_initial_state(tf_env.batch_size)

	timed_at_step = global_step.numpy()
	time_acc = 0

	# Dataset generates trajectories with shape [Bx2x...]
	dataset = replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=config.batch_size,
			num_steps=2).prefetch(3)
	iterator = iter(dataset)
	
	for _ in tqdm.tnrange(config.num_iterations):
		start_time = time.time()
		time_step, policy_state = collect_driver.run(
				time_step=time_step,
				policy_state=policy_state,
		)
		
		experience, _ = next(iterator)
		train_loss = tf_agent.train(experience)
		time_acc += time.time() - start_time

		if global_step.numpy() % config.log_interval == 0:
			tf.logging.info('step = %d, loss = %f', global_step.numpy(), train_loss.loss)
			steps_per_sec = config.log_interval / time_acc
			tf.logging.info('%.3f steps/sec', steps_per_sec)
			time_acc = 0

		if global_step.numpy() % config.eval_interval == 0:
			start = time.time()
			tf.logging.info('step = %d' % global_step.numpy())
			for dist_thresh in [0.1, 0.2, 0.5]:
				tf.logging.info("\t distance threshold = {}".format(dist_thresh))
				eval_tf_env.pyenv.envs[0].gym._set_distance_threshold(dist_thresh)

				results = metric_utils.eager_compute(
						eval_metrics,
						eval_tf_env,
						eval_policy,
						num_episodes=config.num_eval_episodes,
						train_step=global_step,
						summary_prefix='Metrics',
				)
				for (key, value) in results.items():
					tf.logging.info('\t\t %s = %.2f', key, value.numpy())
				# For debugging, it's helpful to check the predicted distances for
				# goals of known distance.
				pred_dist = []
				for _ in range(config.num_eval_episodes):
					ts = eval_tf_env.reset()
					dist_to_goal = tf_agent._get_dist_to_goal(ts)[0]
					pred_dist.append(dist_to_goal.numpy())
				tf.logging.info('\t\t predicted_dist = %.1f (%.1f)' % (np.mean(pred_dist), np.std(pred_dist)))
			tf.logging.info('\t eval_time = %.2f' % (time.time() - start))
				
	return train_loss

def train_goal(config):
	tf.reset_default_graph()
	tf_env = env_load_fn(config)
	eval_tf_env = env_load_fn(config)
	
	agent = UvfAgent(tf_env.time_step_spec(), tf_env.action_spec(),
		max_episode_steps=config.max_episode_steps, 
		use_distributional_rl=config.use_distributional_rl, ensemble_size=config.ensemble_size)

	train_eval(agent, tf_env, eval_tf_env, config)


if __name__ == '__main__':
	config = argparser()
	if(config.experiment == 'env'):
		test_env(config)
	elif(config.experiment == 'goal'):
		train_goal(config)
	elif(config.experiment == 'sorb'):
		train_sorb(config)