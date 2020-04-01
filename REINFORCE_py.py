import gym
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Default parameters for plots
plt.rcParams['font.size'] = 18
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = [9,7]
plt.rcParams['font.family'] = ['Kaiti']
plt.rcParams['axes.unicode_minus'] = False

# Hyperparameter
class Policy(nn.Module):
    def __init__(self,input_size,output_size):
        super(Policy,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,output_size)
        )
        self.memory_list = []

    def forward(self,inputs,training = None):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs = inputs.unsqueeze(0)
        output = nn.Softmax(self.net(inputs),dim=1)
        return output

    def save_memory(self,transition):
        self.memory_list.append(transition)

    def train_net(self,gamma,optimizer,loss_list):
        R = 0.0
        for reward,prob in self.memory_list[::-1]:
            R =  reward + gamma*R
            loss = -torch.log(prob)*R
            loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.memory_list = [] # 轨迹清零

def plot_curse(self,target_list,loss_list):
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