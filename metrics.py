from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tf_agents.metrics import tf_metric
from tf_agents.utils import common
from tf_agents.metrics.tf_metrics import TFDeque

class AvgMaxRewardMetric(tf_metric.TFStepMetric):
  """Metric to compute the average return."""

  def __init__(self,
               name='AvgMaxReward',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AvgMaxRewardMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._max_reward_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')
    self.min_reward = -9999999

  @common.function(autograph=True)
  def call(self, trajectory):
    # Assign least value when the episode starts
    self._max_reward_accumulator.assign(
        tf.where(trajectory.is_first(), tf.constant(value=self.min_reward, shape=tf.shape(self._max_reward_accumulator)),
                 self._max_reward_accumulator))

    # Update accumulator with received rewards.
    def f1(): return trajectory.reward
    def f2(): return self._max_reward_accumulator
    self._max_reward_accumulator.assign(tf.cond(tf.math.greater(trajectory.reward, self._max_reward_accumulator), f1, f2))
    #self._max_reward_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._max_reward_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._max_reward_accumulator.assign(tf.zeros_like(self._max_reward_accumulator))