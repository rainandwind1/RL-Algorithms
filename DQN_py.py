import torch
from  torch import optim,nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

# Hyperparameters
# Replay_time = 30

class DDQN(nn.Module):
    # 动作值函数网络
    def __init__(self,input_size,output_size,memory_len):
        super(DDQN,self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size,128,bias=True),
            nn.ReLU(),
            nn.Linear(128,256,bias=True),
            nn.ReLU(),
            nn.Linear(256,256,bias=True),
            nn.ReLU(),
            nn.Linear(256,self.output_size,bias=True)
        )
        for i in self.net:
            if isinstance(i,nn.Linear):
                nn.init.kaiming_normal_(i.weight)
                nn.init.constant_(i.bias,0.1)
                print("init successful!")
        self.memory_len = memory_len
        self.memory_list = collections.deque(maxlen=memory_len)

    def forward(self,inputs,training=None):
        output = self.net(inputs)
        return output

    def sample_action(self,state,epsilon):
        input = torch.tensor(state,dtype=torch.float32)
        input = input.squeeze(0)
        action_value = self(input)
        coin = np.random.uniform()
        if coin > epsilon:
            action = int(torch.argmax(action_value))
            return action
        else:
            return np.random.randint(0,self.output_size)
        
    # 经验回放部分
    def save_memory(self,transition):
        self.memory_list.append(transition)

    def sample_memory(self,n):
        batch = random.sample(self.memory_list,n)
        s_ls,a_ls,r_ls,s_next_ls,done_mask_ls = [],[],[],[],[]
        for trans in batch:
            s,a,r,s_next,done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_mask_ls.append([done_flag])
        return torch.tensor(s_ls,dtype=torch.float32),\
            torch.tensor(a_ls,dtype=torch.int64),\
            torch.tensor(r_ls,dtype=torch.float32),\
            torch.tensor(s_next_ls,dtype=torch.float32),\
            torch.tensor(done_mask_ls,dtype=torch.float32)


# 训练函数
def train(q_net,q_target,batch_size,gamma,learning_rate,loss_list,Replay_time):
    optimizer = optim.Adam(q_net.parameters(),lr = learning_rate)
    huber = nn.SmoothL1Loss()
    for i in range(Replay_time):
        s,a,r,s_next,done_flag = q_net.sample_memory(batch_size)
        # Q_value
        qa_out = q_net(s)
        a_index = torch.LongTensor(a)
        q_a = torch.gather(qa_out,1,a_index)


        # Q_target value
        # a_target = torch.argmax(qa_out,dim = 1)
        # a_target = torch.reshape(torch.LongTensor(a_target),shape=(batch_size,1))
        # a_target_index = torch.LongTensor(a_target)
        # qtarget_out = q_target(s_next).detach()
        # qtarget_out = torch.gather(qtarget_out,1,a_target_index)
        qtarget_out = q_target(s_next).detach()
        qtarget_out = torch.max(qtarget_out,dim=1)


        q_t = r + gamma*qtarget_out*done_flag

        # 损失与优化
        loss = huber(q_a,q_t)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# 绘制结果
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

# Q = DDQN(4,2,1000)
# for name, param in Q.named_parameters():
#     print(name)