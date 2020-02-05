import os
import gym
import numpy as np

import sac.core as core
from sac.sac import sac
from config import argparser
from sac.utils.run_utils import setup_logger_kwargs
from sac.utils.test_policy import load_policy, run_policy

def sac_run(config):
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
	env = gym.make('FetchReach-v1')
	obs = env.reset()
	step, num_steps, done = 0, 1000, False
	while step < num_steps and done != True:
		obs_dict, rew, done, info = env.step(env.action_space.sample())
		print("Observation : {} - Desired Goal : {} - Achieved Goal : {}".format(obs_dict['observation'], obs_dict['desired_goal'], obs_dict['achieved_goal']))
		env.render()
		step += 1

if __name__ == '__main__':
	config = argparser()
	main(config)	
