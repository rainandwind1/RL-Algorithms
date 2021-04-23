import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from torch.distributions import Categorical
import gym

class TD3(nn.Module):
    def __init__(self, args):
        super(TD3, self).__init__()
        self.input_size, self.output_size, self.action_min, self.action_max, self.clamp, self.lr, self.device = args
        self.toi = 0.02
        self.mem_size = 30000
        self.buffer = ReplayBuffer_ddpg(args = (self.mem_size, self.device))
        
        self.actor_net = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        
        self.critic_net1 = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.critic_net2 = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.target_actor = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        self.target_critic1 = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.target_critic2 = nn.Sequential(
            nn.Linear(self.input_size + self.output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.target_actor.load_state_dict(self.actor_net.state_dict())
        self.target_critic1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic2.load_state_dict(self.critic_net2.state_dict())
        self.optimizer = optim.Adam([{'params':self.actor_net.parameters()}, {'params':self.critic_net1.parameters()}, {'params':self.critic_net2.parameters()}], self.lr)

    def get_tar_action(self, inputs, vec = False):
        raw_op = self.target_actor(inputs)
        noise = torch.rand_like(raw_op).to(self.device)
        action_vec = raw_op
        action_vec = torch.tanh(action_vec) + noise
        if self.clamp:
            action_vec = torch.clamp(action_vec, self.action_min, self.action_max)
        if not vec:
            return action_vec.detach().cpu().numpy()
        else:
            return action_vec

    def get_action(self, inputs, vec = False, eval_mode = False):
        raw_op = self.actor_net(inputs)
        noise = torch.rand_like(raw_op).to(self.device) if not eval_mode else 0.
        action_vec = raw_op
        action_vec = torch.tanh(action_vec) + noise
        if self.clamp:
            action_vec = torch.clamp(action_vec, self.action_min, self.action_max)
        if not vec:
            return action_vec.detach().cpu().numpy()
        else:
            return action_vec

    def update_target(self):
        for raw, target in zip(self.actor_net.parameters(), self.target_actor.parameters()):
            target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)
        
        for raw, target in zip(self.critic_net1.parameters(), self.target_critic1.parameters()):
            target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

        for raw, target in zip(self.critic_net2.parameters(), self.target_critic2.parameters()):
            target.data.copy_(self.toi * raw.data + (1 - self.toi) * target.data)

    
    def train(self, step, gamma = 0.98, batch_size = 32):
        s, a, r, s_next, done = self.buffer.sample_batch(batch_size)
        
        if step % 10 == 0:
            action_vec = self.get_action(s, vec = True)
            policy_loss = -self.critic_net1(torch.cat([s, action_vec], -1)).mean()
            self.update_target()
        else:
            policy_loss = 0.
        
        action_next = self.get_tar_action(s_next, vec=True).detach()
        q_target = r + gamma * torch.min(self.target_critic1(torch.cat([s_next, action_next], -1)), self.target_critic2(torch.cat([s_next, action_next], -1))) * (1 - done)
        q_val1 = self.critic_net1(torch.cat([s, a], -1))
        q_val2 = self.critic_net2(torch.cat([s, a], -1))
        # q_target = r + gamma * self.target_critic1(torch.cat([s_next, action_next], -1)) * (1 - done)
        # q_val = self.critic_net1(torch.cat([s, a], -1))
        critic_loss = (q_target.detach() - q_val1) ** 2 + (q_target.detach() - q_val2) ** 2

        loss = policy_loss.mean() + critic_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()

def test_TD3():
    env = gym.make("Pendulum-v0")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TD3(args = (3, 2, -2, 2, True, 1e-3, device)).to(device)
    total_step = 0
    for i in range(10000):
        s = env.reset()
        score = 0.
        for t in range(200):
            total_step += 1
            action = model.get_action(torch.FloatTensor(s).to(device))

            s_next, reward, done, info = env.step(action)
            model.buffer.save_trans((s, action, reward, s_next, done))
            score += reward 
            s = s_next
            if len(model.buffer.buffer) > 60:
                model.train(total_step)
            if done:
                break
        print("Epoch:{}    Score:{}".format(i+1, score))

