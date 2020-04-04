import torch
from  torch import optim,nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import gym




# Hyperparameters
# Replay_time = 30

class Replay_buffer():
    def __init__(self, alpha, beta, memory_len):
        self.memory_buffer = collections.deque(maxlen=memory_len)
        self.alpha = alpha
        self.beta = beta
        self.memory_len = memory_len
        self.index = 0
        self.priority = np.zeros([self.memory_len], dtype=np.float32)

    def store_transition(self,transition):
        max_prior = np.max(self.priority) if self.memory_buffer else 1.0
        self.memory_buffer.append(transition)    # transition : S_t , A_s, R_t, S_t+1, done_flag
        self.priority[self.index] = max_prior
        self.index += 1
        self.index %= self.memory_len
        
    def replay_memory(self,batch_size):
        if len(self.memory_buffer) < self.memory_len:
            probs = self.priority[:len(self.memory_buffer)]
        else:
            probs = self.priority
        probs = probs**self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory_buffer), batch_size, p=probs)
        samples = [self.memory_buffer[idx] for idx in indices]
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        for trans in samples:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        s_ls = torch.tensor(s_ls, dtype=torch.float32)
        a_ls = torch.tensor(a_ls, dtype=torch.int64)
        r_ls = torch.tensor(r_ls, dtype=torch.float32)
        s_next_ls = torch.tensor(s_next_ls, dtype=torch.float32)
        done_flag_ls = torch.tensor(done_flag_ls,dtype=torch.float32)
        samples = (s_ls, a_ls, r_ls, s_next_ls, done_flag_ls)

        # 计算重要性采样权重
        weight = (len(self.memory_buffer)*probs[indices])**(-self.beta)
        weight = weight/np.max(weight)
        weight = np.array(weight,dtype=np.float32)

        return weight,indices,samples

    def update_priority(self,indices,priority):
        for index, prior in zip(indices, priority):
            self.priority[index] = prior



    @ property
    def len(self):
        return len(self.memory_buffer)

class DQN(nn.Module):
    # 动作值函数网络
    def __init__(self, input_size, output_size, memory_len):
        super(DQN,self).__init__()
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
        print("Net Weight init successful!")
        self.memory_len = memory_len
        self.memory_list = collections.deque(maxlen=memory_len)

    def forward(self, inputs, training=None):
        output = self.net(inputs)
        return output

    def sample_action(self, state, epsilon):
        input = torch.tensor(state,dtype=torch.float32,requires_grad=True)
        input = input.squeeze(0)
        action_value = self(input)
        coin = np.random.uniform()
        if coin > epsilon:
            action = int(torch.argmax(action_value))
            return action
        else:
            return np.random.randint(0,self.output_size)


# 训练函数
def train(Q_value, Q_target, replay_buffer, optimizer, loss, gamma, loss_list, Replay_time=20):
    for i in range(Replay_time):
        weight_ratio, prior_indices, batch_samples = replay_buffer.replay_memory(batch_size)
        weight_ratio = torch.tensor(weight_ratio,dtype=torch.float32)
        s, a, r, s_next, done_flag = batch_samples
        # Q_value
        a = torch.LongTensor(a)
        q_a = Q_value(s)
        q_a = torch.gather(q_a,1,a)
        # Q_target
        q_target = Q_target(s_next).detach()
        q_target = torch.max(q_target,dim = 1,keepdim=True)[0]
        q_target = r + gamma*q_target*done_flag

        # 缩减维度
        q_target = q_target.squeeze(1)
        q_a = q_a.squeeze(1)
        loss = (q_target - q_a).pow(2)*weight_ratio
        priorities = loss + 1e-5
        replay_buffer.update_priority(prior_indices,priorities)
        loss = loss.mean()
        loss_list.append(loss)
        # 损失与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# 绘制结果
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
    env = env.unwrapped
    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space)
   
    # 超参数设置
    gamma = 0.99
    learning_rate = 0.0001
    output_size = 2
    state_size = 4
    memory_len = 10000
    replay_len = 2000
    epoch_num = 600   # 回合数
    max_steps = 400   # 最大步数
    update_target_interval = 50 # 目标网络更新间隔
    batch_size = 64
    train_flag = False
    alpha = 0.6
    beta = 0.4    
    score_list = []
    loss_list = []
    
    # 初始化
    Q_value = DQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
    Q_target =  DQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
    replay_buff = Replay_buffer(alpha,beta,memory_len)
    huber = nn.SmoothL1Loss()
    optimizer = optim.Adam(Q_value.parameters(),lr = learning_rate)

    
    for i in range(epoch_num):
        epsilon = max(0.01,0.2-0.01*(i)/200)
        s = env.reset()
        score = 0
        for j in range(max_steps):
            env.render()
            a = Q_value.sample_action(s,epsilon=epsilon)
            s_next,reward,done,info = env.step(a)
            done_flag = 0.0 if done else 1.0
            replay_buff.store_transition((s,a,reward/10,s_next,done_flag))
            score += reward
            s = s_next
            if done:
                break
        score_list.append(score)
        if replay_buff.len >= replay_len:
            train_flag = True
            train(Q_value,Q_target,replay_buff,optimizer,huber,gamma,loss_list,Replay_time=20)
        # 更新目标网络
        if (i+1) % update_target_interval == 0 and i > 0:
            Q_target.load_state_dict(Q_value.state_dict())
            print("Target net load weight !")
        print(" {} epoch:   score: {}   training: {}".format(i+1,score,train_flag))
    plot_curse(score_list,loss_list)
    env.close()


