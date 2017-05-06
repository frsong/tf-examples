"""
Asynchronous advantage actor-critic based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import logging
import numpy as np
import cv2

# OpenAI Gym/Universe modules
from gym.spaces.box import Box
import gym
import universe
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._log_interval    = log_interval
        self._episode_reward  = 0
        self._episode_length  = 0
        self._all_rewards     = []

    def _after_reset(self, obs):
        logger.info("Resetting environment")
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards    = []
        return obs

    def _after_step(self, obs, reward, done, info):
        to_log = {}

        if reward is not None:
            self._episode_reward += reward
            if obs is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info(
                "Episode terminating: episode_reward = %s episode_length = %s",
                self._episode_reward, self._episode_length
                )
            to_log['global/episode_reward'] = self._episode_reward
            to_log['global/episode_length'] = self._episode_length
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards    = []

        return obs, reward, done, to_log

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

def _process_frame(frame):
    """
    After cropping, resize by half, then down to 42x42
    (essentially mipmapping). If we resize directly we lose pixels that,
    when mapped to 42x42, aren't close enough to the pixel boundary.

    """
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class Rescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(Rescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame(observation) for observation in observation_n]

def create_atari_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)

    # Process the environment
    env = Vectorize(env)
    env = Rescale(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    return env

def create_env(env_id, seed):
    return create_atari_env(env_id, seed)
