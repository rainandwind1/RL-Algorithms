import torch
import gym
from torch import nn, optim
import random
import collections
import numpy as np


if __name__ == "__main__":
    
    MAX_EPOCH = 10000

    env = gym.make('LunarLander-v2')
    env.seed(0)

    print('State shape:', env.observation_space.shape)
    print('Number of actions:', env.action_space.n)

    state = env.reset()
    for epo_i in range(MAX_EPOCH):
        done = False
        while not done:
            action = random.sample(range(env.action_space.n), 1)[0]
            env.render()
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
    env.close()