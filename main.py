import os
import gym
import numpy as np
from config import argparser
from environment import env_load_fn
from tf_agents.policies import random_tf_policy

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

def main(config):
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
if __name__ == '__main__':
	config = argparser()
	main(config)	
