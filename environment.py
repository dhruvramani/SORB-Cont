import gym
import gym.spaces
import numpy as np
import base64
import imageio

from tf_agents.environments import suite_gym
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from envs.fetch.reach import FetchReachEnv
from envs.fetch.push import FetchPushEnv
from envs.fetch.slide import FetchSlideEnv
from envs.fetch.pick_and_place import FetchPickAndPlaceEnv

class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
    """Resets the environment without setting done = True.

    Resets the environment if either these conditions holds:
        1. The base environment returns done = True
        2. The time limit is exceeded.
    """

    def __init__(self, env, duration):
        super(NonTerminatingTimeLimit, self).__init__(env)
        self._duration = duration
        self._step_count = None

    def _reset(self):
        self._step_count = 0
        return self._env.reset()

    @property
    def duration(self):
        return self._duration

    def _step(self, action):
        if self._step_count is None:
            return self.reset()

        ts = self._env.step(action)

        self._step_count += 1
        if self._step_count >= self._duration or ts.is_last():
            self._step_count = None

        return ts

def create_policy_eval_video(tf_env, policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = tf_env.reset()
      video.append_data(tf_env.pyenv.envs[0].gym.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        video.append_data(tf_env.pyenv.envs[0].gym.render())
  return True


def env_load_fn(config, gym_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    obs = ([grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel])

    Args:
        environment_name: Name for the environment to load.
        max_episode_steps: If None the max_episode_steps will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no timestep_limit set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        terminate_on_timeout: Whether to set done = True when the max episode
            steps is reached.

    Returns:
        A PyEnvironmentBase instance.
    """
    envs_map = {'FetchReach-v1' : FetchReachEnv, 'FetchPush-v1' : FetchPushEnv,
             'FetchPickAndPlace-v1' : FetchPickAndPlaceEnv, 'FetchSlide-v1' : FetchSlideEnv}
    gym_env = envs_map[config.env_name](reward_type=config.reward_type) #suite_gym.load(environment_name)
    
    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)
    
    env = gym_wrapper.GymWrapper(gym_env, discount=1.0, auto_reset=True)
   
    if config.max_episode_steps > 0:
        if config.terminate_on_timeout:
            env = wrappers.TimeLimit(env, config.max_episode_steps)
        else:
            env = NonTerminatingTimeLimit(env, config.max_episode_steps)

    return tf_py_environment.TFPyEnvironment(env)
