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
from td3_uvf_agent import Td3UvfAgent
from train import train_eval, td3_train_eval
from search_policy import SearchPolicy

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

def train_td3(config):
	tf.reset_default_graph()
	tf_env = env_load_fn(config)
	eval_tf_env = env_load_fn(config)

	global_step = tf.train.get_or_create_global_step()
	agent = Td3UvfAgent(tf_env.time_step_spec(), tf_env.action_spec(),
		train_step_counter=global_step)

	td3_train_eval(agent, tf_env, eval_tf_env, config)
	#create_policy_eval_video(eval_tf_env, agent.policy, './td3-agent')

# ---------- SORB Manipulation ----------

def train_uvf(config):
	tf.reset_default_graph()
	tf_env = env_load_fn(config)
	eval_tf_env = env_load_fn(config)
	
	agent = UvfAgent(tf_env.time_step_spec(), tf_env.action_spec(),
		max_episode_steps=config.max_episode_steps, 
		use_distributional_rl=config.use_distributional_rl, ensemble_size=config.ensemble_size)

	train_eval(agent, tf_env, eval_tf_env, config)
	#create_policy_eval_video(eval_tf_env, agent.policy, './uvf-agent')

def train_sorb(config):
	tf.reset_default_graph()
	tf_env = env_load_fn(config)
	eval_tf_env = env_load_fn(config)

	agent = UvfAgent(tf_env.time_step_spec(), tf_env.action_spec(),
		max_episode_steps=config.max_episode_steps, 
		use_distributional_rl=config.use_distributional_rl, ensemble_size=config.ensemble_size)

	train_eval(agent, tf_env, eval_tf_env, config)
	rb_vec = []
	for _ in tqdm.tnrange(config.replay_buffer_size):
  		time_step = eval_tf_env.reset()
  		rb_vec.append(time_step.observation['observation'].numpy()[0])
  	rb_vec = np.array(rb_vec)

  	pdist = agent._get_pairwise_dist(rb_vec, aggregate=None).numpy()



if __name__ == '__main__':
	config = argparser()
	if(config.experiment == 'env'):
		test_env(config)
	elif(config.experiment == 'td3'):
		train_td3(config)
	elif(config.experiment == 'uvf'):
		train_uvf(config)
	elif(config.experiment == 'sorb'):
		train_sorb(config)
