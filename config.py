import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def sac_parser(parser):
    # ---- SAC ----
    parser.add_argument('--env', type=str, default='Reacher-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='reacher_sac')
    parser.add_argument('--save_dir', type=str, default='./models/')

    # SAC Testing
    parser.add_argument('--test_sac', type=str2bool, default='True')
    parser.add_argument('--test_len', '-l', type=int, default=0)
    parser.add_argument('--test_episodes', '-n', type=int, default=100)
    parser.add_argument('--test_render', '-nr', type=str2bool, default='True')
    parser.add_argument('--test_itr', '-i', type=int, default=-1)
    parser.add_argument('--test_deterministic', '-d', action='store_true')

    return parser

def argparser():
    parser = argparse.ArgumentParser("SORB for manupilation tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser = sac_parser(parser)
    parser.add_argument('--experiment', type=str, default='goal', choices=['env', 'goal', 'sorb'])
    parser.add_argument('--env_name', type=str, default='FetchReach-v1')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=10000) # 20
    parser.add_argument('--use_distributional_rl', type=str2bool, default='True')
    parser.add_argument('--ensemble_size', type=int, default=3)
 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--initial_collect_steps', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--num_iterations', type=int, default=30000) # 100,000
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1000)


    args = parser.parse_args()
    return args
