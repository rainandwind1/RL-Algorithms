import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from torch.distributions import Categorical
import gym

class DDPG(nn.Module):
    def __init__(self, args):
        super(DDPG, self).__init__()
        self.input_size, self.output_size, self.action_min, self.action_high, self.clamp, self.lr, self.device = args
        self.mem_size = 30000
        self.buffer = ReplayBuffer_ddpg(args = (self.mem_size, self. device))
        self.toi = 0.02

        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        self.target_actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.target_critic = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optimizer = optim.Adam([{'params':self.actor.parameters()}, {'params':self.critic.parameters()}], self.lr)

    def get_tar_action(self, inputs, vec = False, eval_mode = False):
        raw_op = self.target_actor(inputs)
        noise = torch.rand_like(raw_op).to(self.device) if not eval_mode else 0.
        action_vec = raw_op + noise
        action_vec = torch.tanh(action_vec)
        if self.clamp:
            action_vec = torch.clamp(action_vec, self.action_min, self.action_high)
        if vec:
            return action_vec
        else:
            return action_vec.detach().cpu().numpy()

    def get_action(self, inputs, vec = False, eval_mode = False):
        raw_op = self.actor(inputs)
        noise = torch.rand_like(raw_op).to(self.device) if not eval_mode else 0.
        action_vec = raw_op + noise
        action_vec = torch.tanh(action_vec)
        if self.clamp:
            action_vec = torch.clamp(action_vec, self.action_min, self.action_high)
        if vec:
            return action_vec
        else:
            return action_vec.detach().cpu().numpy()
        
    def train(self, gamma = 0.99, batch_size = 32):
        s, a, r, s_next, done = self.buffer.sample_batch(batch_size)
        
        action_vec = self.get_action(s, vec = True)
        policy_loss = -self.critic(torch.cat([s, action_vec], -1))
        
        q_target = r + gamma * self.target_critic(torch.cat([s_next, self.get_tar_action(s_next, vec=True)], -1)) * (1 - done)
        q_val = self.critic(torch.cat([s, a], -1))
        critic_loss = (q_target.detach() - q_val) ** 2
        
        loss = critic_loss.mean() + policy_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target() 

    def update_target(self):
        
        for raw, target in zip(self.actor.parameters(), self.target_actor.parameters()):
            target.data = self.toi * raw.data + (1 - self.toi) * target.data
        
        for raw, target in zip(self.critic.parameters(), self.target_critic.parameters()):
            target.data = self.toi * raw.data + (1 - self.toi) * target.data

            
            
def test_ddpg():
    env = gym.make("Pendulum-v0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DDPG(args = (3, 2, -2, 2, True, 1e-3, device)).to(device)
    for i in range(10000):
        s = env.reset()
        score = 0.
        for t in range(200):
            action = model.get_action(torch.FloatTensor(s).to(device))

            s_next, reward, done, info = env.step(action)
            model.buffer.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if len(model.buffer.buffer) > 60:
                model.train()
            if done:
                break
        print("Epoch:{}    Score:{}".format(i+1, score))
