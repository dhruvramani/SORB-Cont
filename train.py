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

def td3_train_eval(tf_agent, tf_env, eval_tf_env, config):
    """A simple train and eval for UVF.  """
    
    tf.logging.info('random_seed = %d' % config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    tf.set_random_seed(config.random_seed)
    
    #max_episode_steps = tf_env.pyenv.envs[0]._duration
    global_step = tf.train.get_or_create_global_step()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size)

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=config.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=config.num_eval_episodes)
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

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)
    
    for _ in tqdm.tnrange(config.num_iterations):
        start_time = time.time()
        time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
        )

        for _ in range(config.train_steps_per_iteration):
            train_loss = train_step()

        time_acc += time.time() - start_time

        if global_step.numpy() % config.log_interval == 0:
            tf.logging.info('step = %d, loss = %f', global_step.numpy(), train_loss.loss)
            steps_per_sec = config.log_interval / time_acc
            tf.logging.info('%.3f steps/sec', steps_per_sec)
            time_acc = 0

        if global_step.numpy() % config.eval_interval == 0:
            start = time.time()
            tf.logging.info('step = %d' % global_step.numpy())
            for dist_thresh in [0.1, 0.5, 1.0]:
                tf.logging.info("\t distance threshold = {}".format(dist_thresh))
                eval_tf_env.pyenv.envs[0].gym._set_distance_threshold(dist_thresh)
                eval_tf_env.reset()

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
            tf.logging.info('\t eval_time = %.2f' % (time.time() - start))
                
    return train_loss
