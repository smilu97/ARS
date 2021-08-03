#!/usr/bin/env python

import gym
import numpy as np

from ars.linear_policy import LinearPolicy
from ars.config import ARSConfig
from ars.env_ars import EnvARS

def main():
  def env_creator():
    return gym.make('Humanoid-v2')

  env = env_creator()
  input_size = env.observation_space.shape[0]
  output_size = env.action_space.shape[0]

  print('input size:', input_size)
  print('output size:', output_size)

  policy = LinearPolicy(input_size, output_size, discrete_output=False)
  config = ARSConfig(
    step_size=0.1,
    num_directions=100,
    num_top_directions=10,
    exploration_noise=0.1
  )
  
  ars = EnvARS(env_creator, policy, config)

  while True:
    score = ars.evaluate()
    print('score:', score)
    if score >= 10000.0:
      break
    for _ in range(1):
      ars.train()
  
  print('params:', ars.params)
  
  while True:
    obs = env.reset()
    done = False
    while not done:
      env.render()
      obs, reward, done, _ = env.step(policy.call(ars.params, obs))

if __name__ == '__main__':
  main()