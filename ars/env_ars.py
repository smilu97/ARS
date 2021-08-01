# author: smilu97

import gym
import numpy as np

from typing import Callable

from ars.config import ARSConfig
from ars.policy import Policy
from ars import ARS

class EnvARS(ARS):

  def __init__(self, env_creator: Callable[[], gym.Env], policy: Policy, config: ARSConfig):
    self.env = env_creator()
    def evaluator(params: np.ndarray):
      obs = self.env.reset()
      done = False
      sum_reward = 0.0

      while not done:
        action = policy.call(params, obs)
        obs, reward, done, info = self.env.step(action)
        sum_reward += reward
      
      return sum_reward

    super().__init__(evaluator, policy.init_params(), config)
