import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random
from collections import deque
# TD3 6个网络


class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super(Actor,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        policy = self.actor(inputs)
        return policy

    def choose_action(self, inputs, dim):
        policy = self(inputs)
        policy = F.softmax(policy, dim=dim)
        m = Categorical(policy)
        if dim == 0:
            action = m.sample().item()
        else:
            action = m.sample()
            action = action.unsqueeze(1)
            action = action.numpy()
        return action


class Critic(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, mem_len):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.critic = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.mem = deque(maxlen=mem_len)

    def forward(self, state, action, dim):
        state = torch.tensor(state, dtype = torch.float32)
        action = torch.tensor(action, dtype = torch.float32)
        if dim == 0:
            action = action.unsqueeze(0)
            inputs = torch.cat((state, action), dim = dim)
        else:
            inputs = torch.cat((state, action), dim = dim)
        value = self.critic(inputs)
        return value

    def save_trans(self, tran):
        self.mem.append(tran)

    def sample_batch(self, batch_size):
        trans_batch = random.sample(self.mem, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls, dtype = torch.float32),\
                torch.tensor(a_ls, dtype = torch.int64),\
                torch.tensor(r_ls, dtype = torch.float32),\
                torch.tensor(s_next_ls, dtype = torch.float32),\
                torch.tensor(done_flag_ls, dtype = torch.float32)


def train(critic, critic_target, actor, actor_target, batch_size, gamma, delay_lock):
    s, a, r, s_next, done_flag = critic.sample_batch(batch_size)
    q_value = critic(s, a, dim = 1)
    action_next = actor_target.choose_action(s_next, dim=1)
    q_target = r + gamma * critic_target(s_next, action_next, dim = 1) * done_flag
    advantage = q_target - q_value
    loss_critic = advantage**2
    loss = loss_critic.mean()
    critic.optimizer.zero_grad()
    # loss.backward(retain_graph = True)
    loss.backward(retain_graph=True)
    critic_net1.optimizer.step()

    # 更新actor网络
    if delay_lock:
        loss_policy = -q_value.mean()
        actor.optimizer.zero_grad()
        loss_policy.backward()
        actor.optimizer.step()





# hyparameter
LEARNING_RATE = 1e-3
MEM_LEN = 30000
D_INTERVAL = 2
TOI = 0.1
GAMMA = 0.95
MAX_EPOCH = 10000
EXPLORATION = 10000
BATCH_SIZE = 32
DELAY_LOCK = False
UPDATE_INTERVAL = 3


if __name__ == "__main__":
    # 网络初始化
    actor_net = Actor(4, 2, LEARNING_RATE)
    actor_net_target = Actor(4, 2, LEARNING_RATE)
    actor_net_target.load_state_dict(actor_net.state_dict())
    critic_net1 = Critic(4, 1, LEARNING_RATE, MEM_LEN)
    critic_net1_target = Critic(4, 1, LEARNING_RATE, MEM_LEN)
    critic_net1_target.load_state_dict(critic_net1.state_dict())
    critic_net2 = Critic(4, 1, LEARNING_RATE, MEM_LEN)
    critic_net2_target = Critic(4, 1, LEARNING_RATE, MEM_LEN)
    critic_net2_target.load_state_dict(critic_net2.state_dict())

    env = gym.make("CartPole-v1")
    total_step = 0
    Train_flag = False
    for epo_i in range(MAX_EPOCH):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            total_step += 1
            action = actor_net.choose_action(obs, dim=0)
            # env.render()
            obs_next, reward, done, info = env.step(action)
            done_flag = 0. if done else 1.
            critic_net1.save_trans((obs, action, reward, obs_next, done_flag))
            critic_net2.save_trans((obs, action, reward, obs_next, done_flag))

            score += reward
            if total_step >= 1000:
                DELAY_LOCK = False
                Train_flag = True
                if total_step % UPDATE_INTERVAL == 0:
                    DELAY_LOCK = True
                action_next = actor_net_target.choose_action(obs_next, dim=0)
                v1 = critic_net1(obs_next, action_next, dim = 0).detach()
                v2 = critic_net2(obs_next, action_next, dim = 0).detach()
                if v1 >= v2:
                    train(critic_net2, critic_net2_target, actor_net, actor_net_target, BATCH_SIZE, GAMMA, DELAY_LOCK)
                else:
                    train(critic_net1, critic_net1_target, actor_net, actor_net_target, BATCH_SIZE, GAMMA, DELAY_LOCK)
                # 更新目标网络（跟随）
                if DELAY_LOCK:
                    for target, online in zip(critic_net1_target.parameters(), critic_net1.parameters()):
                        target = TOI * online + (1 - TOI) * target
                    for target, online in zip(critic_net2_target.parameters(), critic_net2.parameters()):
                        target = TOI * online + (1 - TOI) * target
                    for target, online in zip(actor_net_target.parameters(), actor_net.parameters()):
                        target = TOI * online + (1 - TOI) * target
            obs = obs_next
            if done:
                print("Epoch: {}  score:  {}  training: {}".format(epo_i, score, Train_flag))
                score = 0
    env.close()











