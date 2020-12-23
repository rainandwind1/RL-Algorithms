import torch
import gym
from torch import nn, optim
import random
import collections
import numpy as np


if __name__ == "__main__":
    
    MAX_EPOCH = 10000

    env = gym.make('Pendulum-v0')
    env.seed(0)

    # print('State shape:', env.observation_space.shape)
    # print('Number of actions:', env.action_space.n)

    state = env.reset()
    for epo_i in range(MAX_EPOCH):
        state = env.reset()
        done = False
        while not done:
            action = [random.uniform(-2, 2)]
            env.render()
            state, reward, done, info = env.step(action)
            if done:
                break
                
    env.close()