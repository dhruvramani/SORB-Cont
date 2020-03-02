from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import time
import numpy as np
import tensorflow 
import scipy.sparse.csgraph
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.policies import gaussian_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import time_step

from goal_ac import GoalConditionedActorNetwork, GoalConditionedCriticNetwork

class Td3Info(collections.namedtuple(
    'Td3Info', ('actor_loss', 'critic_loss'))):
  pass

class Td3UvfAgent(tf_agent.TFAgent):
  """A TD3-Uvf Agent."""

    def __init__(self,
               time_step_spec,
               action_spec,
               exploration_noise_std=0.1,
               target_update_tau=0.05,
               target_update_period=5,
               actor_update_period=2,
               dqda_clipping=None,
               td_errors_loss_fn=tf.losses.huber_loss,
               gamma=0.995,
               reward_scale_factor=1.0,
               target_policy_noise=0.2,
               target_policy_noise_clip=0.5,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               max_episode_steps=None,
               ensemble_size=2,
               combine_ensemble_method='min',
               use_distributional_rl=False, 
               train_step_counter=None):
        """Creates a Td3Agent Agent.
        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          actor_network: A tf_agents.network.Network to be used by the agent. The
            network will be called with call(observation, step_type).
          critic_network: A tf_agents.network.Network to be used by the agent. The
            network will be called with call(observation, action, step_type).
          actor_optimizer: The default optimizer to use for the actor network.
          critic_optimizer: The default optimizer to use for the critic network.
          exploration_noise_std: Scale factor on exploration policy noise.
          critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
            the second critic network during Q learning.  The weights from
            `critic_network` are copied if this is not provided.
          target_actor_network: (Optional.)  A `tf_agents.network.Network` to be
            used as the target actor network during Q learning. Every
            `target_update_period` train steps, the weights from `actor_network` are
            copied (possibly withsmoothing via `target_update_tau`) to `
            target_actor_network`.  If `target_actor_network` is not provided, it is
            created by making a copy of `actor_network`, which initializes a new
            network with the same structure and its own layers and weights.
            Performing a `Network.copy` does not work when the network instance
            already has trainable parameters (e.g., has already been built, or when
            the network is sharing layers with another).  In these cases, it is up
            to you to build a copy having weights that are not shared with the
            original `actor_network`, so that this can be used as a target network.
            If you provide a `target_actor_network` that shares any weights with
            `actor_network`, a warning will be logged but no exception is thrown.
          target_critic_network: (Optional.) Similar network as target_actor_network
            but for the critic_network. See documentation for target_actor_network.
          target_critic_network_2: (Optional.) Similar network as
            target_actor_network but for the critic_network_2. See documentation for
            target_actor_network. Will only be used if 'critic_network_2' is also
            specified.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          actor_update_period: Period for the optimization step on actor network.
          dqda_clipping: A scalar or float clips the gradient dqda element-wise
            between [-dqda_clipping, dqda_clipping]. Default is None representing no
            clippiing.
          td_errors_loss_fn:  A function for computing the TD errors loss. If None,
            a default value of elementwise huber_loss is used.
          gamma: A discount factor for future rewards.
          reward_scale_factor: Multiplicative scale for the reward.
          target_policy_noise: Scale factor on target action noise
          target_policy_noise_clip: Value to clip noise.
          gradient_clipping: Norm length to clip gradients.
          debug_summaries: A bool to gather debug summaries.
          summarize_grads_and_vars: If True, gradient and network variable summaries
            will be written during training.
          train_step_counter: An optional counter to increment every time the train
            op is run.  Defaults to the global_step.
          name: The name of this agent. All variables in this module will fall
            under that name. Defaults to the class name.
        """
        tf.Module.__init__(self, name='Td3UvfAgent')

        assert max_episode_steps is not None
        self._max_episode_steps = max_episode_steps
        self._ensemble_size = ensemble_size
        self._use_distributional_rl = use_distributional_rl

        # Create the actor
        self._actor_network = GoalConditionedActorNetwork(time_step_spec.observation, action_spec)
        self._actor_network.create_variables()

        self._target_actor_network = self._actor_network.copy(name='TargetActorNetwork')

        # Create a prototypical critic, which we will copy to create the ensemble.
        critic_net_input_specs = (time_step_spec.observation, action_spec)
        critic_network = GoalConditionedCriticNetwork(
                critic_net_input_specs,
                output_dim=max_episode_steps if use_distributional_rl else None,
        )

        self._critic_network_list = []
        self._target_critic_network_list = []
        for ensemble_index in range(self._ensemble_size):
            self._critic_network_list.append(
                    critic_network.copy(name='CriticNetwork%d' % ensemble_index))
            self._target_critic_network_list.append(
                    critic_network.copy(name='TargetCriticNetwork%d' % ensemble_index))
            self._critic_network_list[-1].create_variables()
            self._target_critic_network_list[-1].create_variables()


        self._actor_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self._critic_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        self._exploration_noise_std = exploration_noise_std
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._actor_update_period = actor_update_period
        self._dqda_clipping = dqda_clipping
        self._td_errors_loss_fn = (
            td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._target_policy_noise = target_policy_noise
        self._target_policy_noise_clip = target_policy_noise_clip
        self._gradient_clipping = gradient_clipping

        self._update_target = self._get_target_updater(
                target_update_tau, target_update_period)

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)

        collect_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)

        collect_policy = gaussian_policy.GaussianPolicy(
            collect_policy,
            scale=self._exploration_noise_std,
            clip=True)

        super(Td3UvfAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=2 if not self._actor_network.state_spec else None,
            train_step_counter=train_step_counter,
            summarize_grads_and_vars=summarize_grads_and_vars)

    def _get_critic_output(self, critic_net_list, next_time_steps, actions=None):
        """Calls the critic net.

        Args:
            critic_net_list: (list) List of critic networks.
            next_time_steps: time_steps holding the observations and step types
            actions: (optional) actions to compute the Q values for. If None, returns
            the Q values for the best action.
        Returns:
            q_values_list: (list) List containing a tensor of q values for each member
            of the ensemble. For distributional RL, computes the expectation over the
            distribution.
        """
        q_values_list = []
        critic_net_input = (next_time_steps.observation, actions)
        for critic_index in range(self._ensemble_size):
            critic_net = critic_net_list[critic_index]
            q_values, _ = critic_net(critic_net_input, next_time_steps.step_type)
            q_values_list.append(q_values)
        return q_values_list

    def _get_expected_q_values(self, next_time_steps, actions=None):
        if actions is None:
            actions, _ = self._actor_network(next_time_steps.observation, next_time_steps.step_type)

        q_values_list = self._get_critic_output(self._critic_network_list, next_time_steps, actions)

        expected_q_values_list = []
        for q_values in q_values_list:
            if self._use_distributional_rl:
                q_probs = tf.nn.softmax(q_values, axis=1)
                batch_size = q_probs.shape[0]
                bin_range = tf.range(1, self._max_episode_steps + 1, dtype=tf.float32)
                ### NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                neg_bin_range = -1.0 * bin_range
                tiled_bin_range = tf.tile(tf.expand_dims(neg_bin_range, 0), [batch_size, 1])
                assert q_probs.shape == tiled_bin_range.shape

                ### Take the inner produce between these two tensors
                expected_q_values = tf.reduce_sum(q_probs * tiled_bin_range, axis=1)
                expected_q_values_list.append(expected_q_values)
            else:
                expected_q_values_list.append(q_values)
        return tf.stack(expected_q_values_list)

    def _get_state_values(self, next_time_steps, actions=None, aggregate='mean'):
        """Computes the value function, averaging across bins (for distributional RL)
        and the ensemble (for bootstrap RL).

        Args:
            next_time_steps: time_steps holding the observations and step types
            actions: actions for which to compute the Q values. If None, uses the
            best actions (i.e., returns the value function).
        Returns:
            state_values: Tensor storing the state values for each sample in the
            batch. These values should all be negative.
        """
        with tf.name_scope('state_values'):
            expected_q_values = self._get_expected_q_values(next_time_steps, actions)
            if aggregate is not None:
                if aggregate == 'mean':
                    expected_q_values = tf.reduce_mean(expected_q_values, axis=0)
                elif aggregate == 'min':
                    expected_q_values = tf.reduce_min(expected_q_values, axis=0)
                else:
                    raise ValueError('Unknown method for combining ensemble: %s' %
                                                     aggregate)

            # @dhruvramani : Changed here
            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            # if not self._use_distributional_rl:
            #     min_q_val = -1.0 * self._max_episode_steps
            #     max_q_val = 0.0
            #     expected_q_values = tf.maximum(expected_q_values, min_q_val)
            #     expected_q_values = tf.minimum(expected_q_values, max_q_val)
            return expected_q_values

    def _initialize(self):
        for ensemble_index in range(self._ensemble_size):
            common.soft_variables_update(
                    self._critic_network_list[ensemble_index].variables,
                    self._target_critic_network_list[ensemble_index].variables,
                    tau=1.0)
        # Caution: actor should only be updated once.
        common.soft_variables_update(
                self._actor_network.variables,
                self._target_actor_network.variables,
                tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws

        Args:
            tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
            period: Step interval at which the target networks are updated.
        Returns:
            An operation that performs a soft update of the target network parameters.
        """
        with tf.name_scope('get_target_updater'):
            def update():  # pylint: disable=missing-docstring
                critic_update_list = []
                for ensemble_index in range(self._ensemble_size):
                    critic_update = common.soft_variables_update(
                            self._critic_network_list[ensemble_index].variables,
                            self._target_critic_network_list[ensemble_index].variables, tau)
                    critic_update_list.append(critic_update)
                actor_update = common.soft_variables_update(
                        self._actor_network.variables,
                        self._target_actor_network.variables, tau)
                return tf.group(critic_update_list + [actor_update])

            return common.Periodically(update, period, 'periodic_update_targets')

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)
        transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]), transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _train(self, experience, weights=None):
        #squeeze_time_dim = not self._actor_network.state_spec
        time_steps, actions, next_time_steps = self._experience_to_transitions(experience)

        critic_vars = []
        for ensemble_index in range(self._ensemble_size):
            critic_net = self._critic_network_list[ensemble_index]
            critic_vars.extend(critic_net.variables)
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_vars
            tape.watch(critic_vars)
            critic_loss = self.critic_loss(time_steps, actions, next_time_steps
                                         weights=weights, training=True)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self._apply_gradients(critic_grads, critic_vars, self._critic_optimizer)

        actor_vars = self._actor_network.variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_vars, ('No trainable actor variables to '
                                             'optimize.')
            tape.watch(actor_vars)
            actor_loss = self.actor_loss(time_steps, weights=weights, training=True)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

        # We only optimize the actor every actor_update_period training steps.
        def optimize_actor():
            actor_grads = tape.gradient(actor_loss, actor_vars)
            return self._apply_gradients(actor_grads, actor_vars,
                                       self._actor_optimizer)

        remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)
        tf.cond(pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)

        self.train_step_counter.assign_add(1)
        self._update_target()

        # TODO(b/124382360): Compute per element TD loss and return in loss_info.
        total_loss = actor_loss + critic_loss

        return tf_agent.LossInfo(total_loss,
                                 Td3Info(actor_loss, critic_loss))

    def _apply_gradients(self, gradients, variables, optimizer):
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = tuple(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
              grads_and_vars, self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(grads_and_vars,
                                              self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,
                                              self.train_step_counter)

        return optimizer.apply_gradients(grads_and_vars)

    def critic_loss(self, time_steps, actions, next_time_steps, weights=None, training=False):
        """Computes the critic loss for TD3 training.
        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.
        Returns:
          critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, next_time_steps.step_type,
                training=training)

          # Add gaussian noise to each action before computing target q values
            def add_noise_to_action(action):  # pylint: disable=missing-docstring
                dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                                scale=self._target_policy_noise * \
                                                tf.ones_like(action))
                noise = dist.sample()
                noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                         self._target_policy_noise_clip)
                return action + noise

            noisy_target_actions = tf.nest.map_structure(add_noise_to_action,
                                                       target_actions)

            # Target q-values are the min of the two networks
            target_q_values = self._get_state_values(next_time_steps, target_actions, aggregate='min')

            td_targets = tf.stop_gradient(
              self._reward_scale_factor * next_time_steps.reward +
              self._gamma * next_time_steps.discount * target_q_values)

            pred_td_targets_all = self._get_critic_output(self._critic_network_list, time_steps, actions)

            if self._debug_summaries:
                tensorflow.summary.histogram(
                    name='td_targets', data=td_targets, step=self.train_step_counter)
                with tf.name_scope('td_targets'):
                    tensorflow.summary.scalar(
                      name='mean',
                      data=tf.reduce_mean(input_tensor=td_targets),
                      step=self.train_step_counter)
                    tensorflow.summary.scalar(
                      name='max',
                      data=tf.reduce_max(input_tensor=td_targets),
                      step=self.train_step_counter)
                    tensorflow.summary.scalar(
                      name='min',
                      data=tf.reduce_min(input_tensor=td_targets),
                      step=self.train_step_counter)

            for td_target_idx in range(self._ensemble_size):
                pred_td_targets = pred_td_targets_all[td_target_idx]
                td_errors = td_targets - pred_td_targets
                with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
                    tensorflow.summary.histogram(
                        name='td_errors', data=td_errors, step=self.train_step_counter)
                    tensorflow.summary.histogram(
                        name='pred_td_targets',
                        data=pred_td_targets,
                        step=self.train_step_counter)
                    with tf.name_scope('td_errors'):
                        tensorflow.summary.scalar(
                          name='mean',
                          data=tf.reduce_mean(input_tensor=td_errors),
                          step=self.train_step_counter)
                        tensorflow.summary.scalar(
                          name='mean_abs',
                          data=tf.reduce_mean(input_tensor=tf.abs(td_errors)),
                          step=self.train_step_counter)
                        tensorflow.summary.scalar(
                          name='max',
                          data=tf.reduce_max(input_tensor=td_errors),
                          step=self.train_step_counter)
                        tensorflow.summary.scalar(
                          name='min',
                          data=tf.reduce_min(input_tensor=td_errors),
                          step=self.train_step_counter)
                    with tf.name_scope('pred_td_targets'):
                        tensorflow.summary.scalar(
                          name='mean',
                          data=tf.reduce_mean(input_tensor=pred_td_targets),
                          step=self.train_step_counter)
                        tensorflow.summary.scalar(
                          name='max',
                          data=tf.reduce_max(input_tensor=pred_td_targets),
                          step=self.train_step_counter)
                        tensorflow.summary.scalar(
                          name='min',
                          data=tf.reduce_min(input_tensor=pred_td_targets),
                          step=self.train_step_counter)

            critic_loss = (sum([self._td_errors_loss_fn(td_targets, pred_td_targets) for pred_td_targets in pred_td_targets_all]))
            if nest_utils.is_batched_nested_tensors(
                time_steps, self.time_step_spec, num_outer_dims=2):
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(input_tensor=critic_loss, axis=1)

            if weights is not None:
                critic_loss *= weights

            return tf.reduce_mean(input_tensor=critic_loss)

    def actor_loss(self, time_steps, weights=None, training=False):
        """Computes the actor_loss for TD3 training.
        Args:
          time_steps: A batch of timesteps.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
          training: Whether this loss is being used for training.
          # TODO(b/124383618): Add an action norm regularizer.
        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            actions, _ = self._actor_network(time_steps.observation,
                                           time_steps.step_type,
                                           training=training)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actions)
            avg_expected_q_values = self._get_state_values(time_steps, actions, aggregate='mean')

            actions = tf.nest.flatten(actions)

        dqdas = tape.gradient([avg_expected_q_values], actions)

        actor_losses = []
        for dqda, action in zip(dqdas, actions):
            if self._dqda_clipping is not None:
                dqda = tf.clip_by_value(dqda, -1 * self._dqda_clipping,
                    self._dqda_clipping)
            loss = common.element_wise_squared_loss(
                tf.stop_gradient(dqda + action), action)
            if nest_utils.is_batched_nested_tensors(
                time_steps, self.time_step_spec, num_outer_dims=2):
                # Sum over the time dimension.
                loss = tf.reduce_sum(loss, axis=1)
            if weights is not None:
                loss *= weights
            loss = tf.reduce_mean(loss)
            actor_losses.append(loss)

        actor_loss = tf.add_n(actor_losses)

        with tf.name_scope('Losses/'):
            tensorflow.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)

        return actor_loss

