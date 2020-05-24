import numpy as np
import torch
from torch import nn,optim
import random
import collections
import matplotlib.pyplot as plt
import gym


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


def train(net,optimizer,loss_fn,gamma,loss_list):
    losses = 0.
    for prob,reward,s,s_next,done_flag in net.memory:
        s_next = torch.tensor(s_next,dtype=torch.float32)
        s_next = s_next.unsqueeze(0)
        s = torch.tensor(s,dtype=torch.float32)
        s = s.unsqueeze(0)
        V_next = reward + gamma*net.critic(s_next)*done_flag
        V_s = net.critic(s)
        A = V_next.detach() - V_s
        loss_a = -torch.log(prob)*A.detach()
        losses += loss_a + 0.5*A**2
    loss_list.append(losses)
    optimizer.zero_grad()
    losses.backward()
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





if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.make("Acrobot-v1")
    # env = gym.make("Breakout-ram-v0")
    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space)

    # 超参数设置
    gamma = 0.95
    learning_rate = 0.001
    output_size = 2
    state_size = 4
    epoch_num = 10000   # 回合数
    max_steps = 400   # 最大步数
    train_flag = False

    # 初始化
    AC = ActorCritic(input_size = state_size,output_size=output_size)
    score_list = []
    loss_list = []
    optimizer = optim.Adam(AC.parameters(),lr = learning_rate)
    huber = nn.MSELoss()



    for i in range(epoch_num):
        epsilon = max(0.01,0.1-0.01*(i)/200)
        s = env.reset()
        score = 0
        for j in range(max_steps):
            # env.render()
            a, a_prob = AC.sample_action(s,epsilon=epsilon)
            s_next,reward,done,info = env.step(a)
            done_flag = 0.0 if done else 1.0
            AC.save_memory((a_prob,reward/100,s,s_next,done_flag))
            score += reward
            s = s_next
            if done:
                train_flag = True
                train(AC,optimizer,huber,gamma,loss_list)
                break
        score_list.append(score)
        print("{} epoch score: {}  training: {}".format(i+1,score,train_flag))
    plot_curse(score_list,loss_list)
    env.close()





