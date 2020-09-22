import torch
from torch import optim,nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import gym
import math


class IQN(nn.Module):
    def __init__(self, args):
        super(IQN, self).__init__()
        self.input_size, self.output_size, self.lr, self.device = args
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.phi = nn.Linear(1, 32, bias = False)
        self.phi_bias = nn.Parameter(torch.rand(32), requires_grad = True)
        self.fc = nn.Linear(32, 64)
        self.full_net = nn.Linear(64, self.output_size)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, inputs, toi_num = 64):
        toi = torch.FloatTensor([[np.random.uniform()] for _ in range(toi_num)]).to(self.device)
        Num = torch.FloatTensor([i for i in range(64)]).to(self.device)
        cos_op = torch.cos(Num * toi * np.pi).unsqueeze(-1)                                        # toi_num * n(64) * 1
        q_val_embed = self.net(inputs).view(inputs.shape[0], -1).unsqueeze(1)
        phi = F.relu(self.phi(cos_op).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)                    # 1 * toi_num * embed_opsize
        z_val_embed = F.relu(self.fc(q_val_embed * phi))
        z_val = self.full_net(z_val_embed).transpose(1, 2)                                           # batch * action_size * toi
        return z_val, toi

    def choose_action(self, state, epsilon, samples_K = 32):
        coin = np.random.rand()
        if coin > epsilon:
            inputs = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            z_val = self(inputs, samples_K)[0]
            z_val = z_val.squeeze(0)
            action = int(torch.argmax(z_val.mean(-1), -1))
        else:
            action = random.choice(range(self.output_size))
        return action


class Replaybuffer():
    def __init__(self, args):
        self.mem_len, self.device = args
        self.buffer = collections.deque(maxlen = self.mem_len)

    def save_memory(self, transition):
        self.buffer.append(transition)

    def sample_memory(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_mask_ls = ([] for i in range(5))
        for trans in sample_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_mask_ls.append([done_flag])
        return torch.tensor(s_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(a_ls,dtype=torch.int64).to(self.device),\
            torch.tensor(r_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(s_next_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(done_mask_ls,dtype=torch.float32).to(self.device)


# 训练函数
def train(z_net, z_target, replaybuffer, batch_size, gamma, N, N_, coef_k = None, rou = None):
    s, a, r, s_next, done_flag = replaybuffer.sample_memory(batch_size)
    z, toi = z_net(s, N)
    z = torch.stack([z[i].index_select(0, a[i]) for i in range(batch_size)]).squeeze(1).unsqueeze(-1)
    
    a_best = z_target(s_next, N_)[0].mean(-1).argmax(-1)
    z_t, toi_t = z_target(s_next, N_)
    z_target = torch.stack([z_t[i].index_select(0, a_best[i]) for i in range(batch_size)]).squeeze(1)
    z_target = (r + gamma * z_target * done_flag).unsqueeze(-2)
    delta_ij = z_target.detach() - z
    
    toi = toi.unsqueeze(0)
    weight = torch.abs(toi - delta_ij.le(0.).float())
    loss = F.smooth_l1_loss(z, z_target.detach())
    loss = torch.mean(weight * loss, 1).mean(1)
    loss = torch.mean(torch.ones_like(r).unsqueeze(-1) * loss)
    z_net.optimizer.zero_grad()
    loss.backward()
    z_net.optimizer.step()


if __name__ == "__main__":

    # test forward
    # device  = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # test_model = IQN((4, 2, 1e-3, device))
    # test_input = torch.rand(32, 4)
    # print(test_model(test_input))

    env = gym.make("CartPole-v1")
    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space)

    # 超参数设置
    gamma = 0.99
    learning_rate = 1e-3
    output_size = 2
    state_size = 4
    memory_len = 10000
    epoch_num = 3000   # 回合数
    max_steps = 400   # 最大步数
    update_target_interval = 20 # 目标网络更新间隔
    batch_size = 32
    train_flag = False
    train_len = 400
    N, N_ = 32, 32
    device  = 'cuda' if torch.cuda.is_available() else 'cpu' 
    epsilon = 0.8 

    # 初始化
    model = IQN((state_size, output_size, learning_rate, device)).to(device)
    target_model = IQN((state_size, output_size, learning_rate, device)).to(device)
    target_model.load_state_dict(model.state_dict())
    replaybuffer = Replaybuffer((memory_len, device))

    for i in range(epoch_num):
        s = env.reset()
        score = 0.
        epsilon = max(0.01, epsilon * 0.999)
        for j in range(max_steps):
            # env.render()
            a = model.choose_action(s, epsilon, samples_K = 32)
            s_next, reward, done, info = env.step(a)
            done_flag = 0.0 if done else 1.0
            replaybuffer.save_memory((s, a, reward, s_next, done_flag))
            score += reward
            s = s_next
            if len(replaybuffer.buffer) >= train_len:
                train_flag = True
                train(model, target_model, replaybuffer, batch_size, gamma, N, N_)
            if done:
                break
        # 更新目标网络
        if (i + 1) % update_target_interval == 0 and i > 0:
            target_model.load_state_dict(model.state_dict())
        print("{} epoch score: {}  training: {}  epsilon:{:.3}".format(i+1, score, train_flag, epsilon))
    env.close()