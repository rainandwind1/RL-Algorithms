import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import collections
import random


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



class Actor(nn.Module):  # DDPG确定性策略，单一输出
    def __init__(self, state_size, action_size, buffer_length):
        super(Actor,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = collections.deque(maxlen=buffer_length)
        self.actor = nn.Sequential(
            nn.Linear(self.state_size,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Tanh()  # 约束 action 的输出
        )

    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs.unsqueeze(0)
        return self.actor(inputs)

    def save_memory(self,transition):   # transition: St, At, Rt, St+1, done_flag
        self.replay_buffer.append(transition)

    def sample_memory(self,batch_size):
        s_list = []
        a_list =[]
        r_list = []
        s_next_list = []
        done_mask_list = []
        trans_batch = random.sample(self.replay_buffer,batch_size)
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_next_list.append(s_next)
            done_mask_list.append([done_flag])
        
        return torch.tensor(s_list,dtype=torch.float32),\
            torch.tensor(a_list,dtype=torch.int64),\
            torch.tensor(r_list,dtype=torch.float32),\
            torch.tensor(s_next_list,dtype=torch.float32),\
            torch.tensor(done_mask_list,dtype=torch.float32)


    # def sample_action(self, obs):
    #     obs = torch.tensor(obs,dtype = torch.float32)
    #     obs.unsqueeze(0)
    #     action_prob = self.actor(obs)
    #     ou_noise = OrnsteinUhlenbeckActionNoise()
    #     ou_ns = torch.tensor([[ou_noise() for i in range(action_prob)]],dtype=torch.float32) # ou noisy在惯性系统中应用效果较好
    #     action_prob += ou_ns
    #     soft_p = nn.Softmax(dim=1)
    #     action_prob = soft_p(action_prob)
    #     action_choice = torch.argmax(action_prob)
    #     return action_prob[action_choice], action_choice  # 注意此时的index = num_st-1 为数组的标签



class Critic(nn.Module):
    def __init__(self,state_size,action_size):  #输入为状态和动作
        super(Critic,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.critic = nn.Sequential(
            nn.Linear(self.state_size + self.action_size,64),
            nn.Linear(64,128),
            nn.Linear(128,1)
        )
    
    def forward(self, state, action, batch=False):
        if not batch:
            action = torch.tensor(action,dtype=torch.float32)
            action = action.unsqueeze(0)
            action = action.unsqueeze(1)
            state = torch.tensor(state,dtype=torch.float32)
            state = inputs.unsqueeze(0)
        inputs = torch.cat((state,action),1)
        return self.critic(inputs)




# class ReplayBuffer():
#     def __init__(self, buffer_size, batch_size):


def train(actor_net,actor_target_net,critic_net,critic_target_net,optimizer,loss_list,batch_size,replay_time = 20, gamma = 0.99, toi = 0.001):
    optimizer_a, optimizer_c = optimizer
    loss_list_a, loss_list_c = loss_list
    for i in range(replay_time): # 经验回放的次数
        # L2 正则
        regularzation_loss = 0
        for param in actor_net.parameters():
            regularzation_loss += torch.sum(abs(param))
        s, a, r, s_next = actor_net.sample_memory(batch_size)
        action_target = actor_target_net(s_next)
        target_y = critic_target_net(s_next,action_target,batch=True)
        y = critic_net(s,a,batch=True)
        
        # online net update param
        critic_loss = torch.mean((target_y-y)**2)
        loss_list_c.append(critic_loss)
        optimizer_c.zero_grad()
        critic_loss.backward()
        optimizer_c.step()


        actor_loss = -torch.mean(y)
        loss_list_a.append(actor_loss)
        optimizer_a.zero_grad()
        actor_loss.backward()
        optimizer_a.step()

        # target net update
        for param_t, param_o in zip(actor_target_net.parameters(), actor_net.parameters()):
            param_t = toi*param_o + (1-toi)*param_t

        for param_t, param_o in zip(critic_target_net.parameters(), critic_net.parameters()):
            param_t = toi*param_o + (1-toi)*param_t





def plot_curse(target_list,loss_list):
    figure1 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(target_list)):
        X.append(i)
    plt.plot(X,target_list,'-r')
    plt.xlabel('epoch')
    plt.ylabel('score')

    figure2 = plt.figure()
    plt.grid()
    Actor_loss = []
    Critic_loss = []
    loss_list_a, loss_list_c = loss_list
    for i in range(len(loss_list)):
        X.append(i)
    plt.plot(X,loss_list_a,'-b')
    plt.plot(X,loss_list_c,'-r')
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.legend(["Actor loss","Critic loss"])
    plt.show()