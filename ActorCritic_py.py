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
        inputs = torch.tensor(inputs,dtype=torch.float32,requires_grad=True)
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
        s_next = torch.tensor(s_next,dtype=torch.float32,requires_grad=True)
        s_next = s_next.unsqueeze(0)
        s = torch.tensor(s,dtype=torch.float32,requires_grad=True)
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





if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.make("Acrobot-v1")
    # env = gym.make("Breakout-ram-v0")
    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space)

    # 超参数设置
    gamma = 0.99
    learning_rate = 0.008
    output_size = 2
    state_size = 4
    memory_len = 10000
    epoch_num = 1200   # 回合数
    max_steps = 400   # 最大步数
    update_target_interval = 50 # 目标网络更新间隔
    batch_size = 64
    train_flag = False
    train_len = 2000

    # 初始化
    Q_value = DDQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
    Q_target =  DDQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
    score_list = []
    loss_list = []
    optimizer = optim.Adam(Q_value.parameters(),lr = learning_rate)
    huber = nn.SmoothL1Loss()



    for i in range(epoch_num):
        epsilon = max(0.01,0.16-0.01*(i)/200)
        s = env.reset()
        score = 0
        for j in range(max_steps):
            env.render()
            a = Q_value.sample_action(s,epsilon=epsilon)
            s_next,reward,done,info = env.step(a)
            done_flag = 0.0 if done else 1.0
            Q_value.save_memory((s,a,reward/100,s_next,done_flag))
            score += reward
            s = s_next
            if done:
                break
        score_list.append(score)
        if len(Q_value.memory_list) >= train_len:
            train_flag = True
            train(Q_value,Q_target,optimizer,huber,batch_size,gamma,loss_list,Replay_time=20)
        # 更新目标网络
        if (i+1) % update_target_interval == 0 and i > 0:
            Q_target.load_state_dict(Q_value.state_dict())
        print("{} epoch score: {}  training: {}".format(i+1,score,train_flag))



    plot_curse(score_list,loss_list)
    env.close()





