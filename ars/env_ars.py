# author: smilu97

import gym
import ray
import numpy as np

from typing import Callable

from ars.config import ARSConfig
from ars.policy import Policy
from ars import ARS

@ray.remote
class EnvActor:
  def __init__(self, env_creator: Callable[[], gym.Env], policy: Policy):
    self.env = env_creator()
    self.policy = policy
  
  def evaluate_n_times(self, params: np.ndarray, mean_state: np.ndarray, var_state: np.ndarray, states: list, n: int=1):
    rewards = []
    for _ in range(n):
      rewards.append(self.evaluate(params, mean_state, var_state, states))
    return np.mean(rewards)
  
  def evaluate(self, params: np.ndarray, mean_state: np.ndarray, var_state: np.ndarray, states: list):
    obs = self.env.reset()
    done = False
    sum_reward = 0.0

    while not done:
      obs = (obs - mean_state) / np.sqrt(var_state)
      action = self.policy.call(params, obs)
      obs, reward, done, info = self.env.step(action)
      sum_reward += reward
      states.append(obs)
    
    return sum_reward

class EnvARS(ARS):

  def __init__(
    self,
    env_creator: Callable[[], gym.Env],
    policy: Policy,
    config: ARSConfig,
    num_cpus: int = 1,
    num_eval_per_param: int=1
  ):

    self.num_cpus = num_cpus

    if not ray.is_initialized():
      ray.init(num_cpus=num_cpus)

    self.actors = [EnvActor.remote(env_creator, policy) for _ in range(num_cpus)]

    len_mean_states = 0
    mean_states = 0
    sqr_mean_states = 0

    def evaluator(params: np.ndarray):
      results = []
      new_states = []
      for i in range(0, len(params), num_cpus):
        sz = min(len(params) - i, num_cpus)
        var_states = sqr_mean_states - np.square(mean_states)

        refs = [self.actors[j].evaluate_n_times.remote(
          params[i+j],
          mean_states,
          var_states,
          new_states,
          num_eval_per_param
        ) for j in range(sz)]

        ray.wait(refs)
        results += [ray.get(x) for x in refs]
      
      # Update mean_states and sqr_mean_states
      new_states = np.array(new_states)
      new_mean_states = np.mean(new_states, axis=0)
      new_sqr_mean_states = np.mean(np.square(new_states), axis=0)

      ratio_prev = (len_mean_states / (len_mean_states + len(new_states)))
      ratio_next = (len(new_states) / (len_mean_states + len(new_states)))
      mean_states = mean_states * raio_prev + new_mean_states * ratio_next
      sqr_mean_states = sqr_mean_states * raio_prev + new_sqr_mean_states * ratio_next
      len_mean_states += len(new_states)

      return results

    super().__init__(evaluator, policy.init_params(), config)
