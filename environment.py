import gym
import gym.spaces
import numpy as np

from tf_agents.environments import suite_gym
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

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

def env_load_fn(environment_name,
                 max_episode_steps=None,
                 gym_env_wrappers=(),
                 terminate_on_timeout=False):
    """Loads the selected environment and wraps it with the specified wrappers.

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
    env = suite_gym.load(environment_name)
    
    for wrapper in gym_env_wrappers:
        env = wrapper(env)
   
    if max_episode_steps > 0:
        if terminate_on_timeout:
            env = wrappers.TimeLimit(env, max_episode_steps)
        else:
            env = NonTerminatingTimeLimit(env, max_episode_steps)

    return tf_py_environment.TFPyEnvironment(env)
