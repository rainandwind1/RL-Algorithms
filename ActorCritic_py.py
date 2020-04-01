import numpy as np
import torch
from torch import nn,optim
import random
import collections
import matplotlib.pyplot as plt



class ActorCritic(nn.Module):
    def __init__(self,input_size,output_size):
        super(ActorCritic,self).__init__()
        self.state_size = input_size
        self.action_size = output_size
        self.actor = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.memory = []

    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs = inputs.unsqueeze(0)
        a_prob = nn.Softmax(self.actor(inputs),dim=1)
        v = self.critic(inputs)
        return a_prob, v

    def save_memory(self,transition):
        self.memory.append(transition)

    def sample_action(self,inputs,epsilon):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs = inputs.unsqueeze(0)
        p = nn.Softmax(dim=1)
        prob = p(self.actor(inputs))
        rand_num = np.random.rand()
        if rand_num > epsilon:
            return int(torch.argmax(prob)),torch.max(prob)
        else:
            action_choice = np.random.choice(self.action_size)
            return action_choice, prob[0][action_choice]


def train(net,optimizer,gamma,loss_list):
    for prob,reward,s,s_next in net.memory:
        s_next = torch.tensor(s_next,dtype=torch.float32)
        s_next = s_next.unsqueeze(0)
        s = torch.tensor(s,dtype=torch.float32)
        s = s.unsqueeze(0)
        V_next = reward + gamma*net.critic(s_next)
        V_s = net.critic(s)
        A = V_next - V_s
        loss_a = -torch.log(prob)*A
        losses = loss_a + 0.5*A**2
        loss_list.append(losses)
        optimizer.zero_grad()
        losses.backward(retain_graph=True)
        optimizer.step()
        net.memory = []



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
    X = []
    for i in range(len(loss_list)):
        X.append(i)
    plt.plot(X,loss_list,'-b')
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.show()

def save_param(model,path):
    targte = model.state_dict()
    torch.save(targte,path)

def load_param(model,path):
    model.load_state_dict(torch.load(path))

