import gym
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import random
# Default parameters for plots
plt.rcParams['font.size'] = 18
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = [9,7]
plt.rcParams['font.family'] = ['Kaiti']
plt.rcParams['axes.unicode_minus'] = False

# Hyperparameter
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy,self).__init__()
        self.state_size = input_size
        self.action_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,output_size),
        )
        self.memory_list = []

    def forward(self, inputs, training = None):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs = inputs.unsqueeze(0)
        output = nn.Softmax(self.net(inputs),dim=1)
        return output

    def sample_action(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        inputs = inputs.unsqueeze(0)
        output = nn.Softmax(dim=1)
        output = output(self.net(inputs))
        a_p = [0. for i in range(self.action_size)]
        for i in range(self.action_size):
            a_p[i] = output[0][i].item()
        p_sum = sum(a_p)
        for i in range(self.action_size):
            a_p[i] = a_p[i]/p_sum
        # print(sum(a_p))
        a = np.random.choice(range(self.action_size),1,p = a_p)[0]
        prob = output[0][a]
        return a, prob

    def save_memory(self, transition):
        self.memory_list.append(transition)

    def train_net(self, gamma, optimizer, loss_list):
        R = 0.0
        for reward,prob in self.memory_list[::-1]:
            R =  reward + gamma*R
            loss = -torch.log(prob)*R
            loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.memory_list = [] # 轨迹清零

def plot_curse(target_list, loss_list):
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

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.make("Acrobot-v1")
    # env = gym.make("Breakout-ram-v0")
    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space)

    # 超参数设置
    gamma = 0.98
    learning_rate = 0.0002
    output_size = 2
    state_size = 4
    epoch_num = 200   # 回合数
    max_steps = 400   # 最大步数
    train_flag = False

    # 初始化
    Agent = Policy(input_size = state_size,output_size = output_size)
    score_list = []
    loss_list = []
    optimizer = optim.Adam(Agent.parameters(),lr = learning_rate)
    huber = nn.MSELoss()



    for i in range(epoch_num):
        s = env.reset()
        score = 0
        for j in range(max_steps):
            env.render()
            a, a_prob = Agent.sample_action(s)
            s_next,reward,done,info = env.step(a)
            done_flag = 0.0 if done else 1.0
            Agent.save_memory((reward/10,a_prob))
            score += reward
            s = s_next
            if done:
                train_flag = True
                Agent.train_net(gamma,optimizer,loss_list)
                break
        score_list.append(score)
        print("{} epoch score: {}  training: {}".format(i+1,score,train_flag))
    plot_curse(score_list,loss_list)
    env.close()

