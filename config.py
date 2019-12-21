import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def argparser():
    parser = argparse.ArgumentParser("SORB for manupilation tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ---- SAC ----
    parser.add_argument('--env', type=str, default='Reacher-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='reacher_sac')
    parser.add_argument('--save_dir', type=str, default='./models/')

    # Policy Testing
    parser.add_argument('--test_sac', type=str2bool, default='True')
    parser.add_argument('--test_len', '-l', type=int, default=0)
    parser.add_argument('--test_episodes', '-n', type=int, default=100)
    parser.add_argument('--test_render', '-nr', type=str2bool, default='True')
    parser.add_argument('--test_itr', '-i', type=int, default=-1)
    parser.add_argument('--test_deterministic', '-d', action='store_true')

    args = parser.parse_args()
    return args
