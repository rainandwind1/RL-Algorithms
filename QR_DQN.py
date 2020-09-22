import torch
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import gym
import math


class Replaybuffer():
    def __init__(self, args):
        self.mem_len, self.device = args
        self.buffer = collections.deque(maxlen = self.mem_len)

    def save_memory(self, transition):
        self.buffer.append(transition)

    def sample_batch(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_mask_ls = ([] for i in range(5))
        for trans in sample_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_mask_ls.append([done_flag])
        return torch.FloatTensor(s_ls).to(self.device),\
            torch.LongTensor(a_ls).to(self.device),\
            torch.FloatTensor(r_ls).to(self.device),\
            torch.FloatTensor(s_next_ls).to(self.device),\
            torch.FloatTensor(done_mask_ls).to(self.device)


class QRDQN(nn.Module):
    def __init__(self, args):
        super(QRDQN, self).__init__()
        self.input_size, self.output_size, self.lr, self.device = args
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.N = 100
        self.toi = torch.arange(0, self.N + 1, device = self.device, dtype = torch.float32) / self.N
        self.toi_hats = ((self.toi[1:] + self.toi[:-1]) / 2.0).view(1, self.N)
        self.embedding_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.N * self.output_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, inputs):
        embed_op = self.net(inputs)
        quantiles = self.embedding_net(embed_op).view(-1, self.N , self.output_size)
        return quantiles

    def choose_action(self, inputs, epsilon):
        inputs = torch.FloatTensor(inputs).to(self.device)
        coin = np.random.rand()
        if coin > epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            quantiles = self(inputs)
            q_val = quantiles.mean(1).squeeze(0)
            return torch.argmax(q_val, -1).item()
        
def cal_huberloss(td_error, kappa = 1.):
    return torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa)
    )

def train(model, target_model, replay_buffer, gamma, batch_size, kappa = 1.):
    s, a, r, s_next, done_flag = replay_buffer.sample_batch(batch_size)
    a = a.unsqueeze(1).expand(batch_size, model.N, 1)
    q_val = model(s)
    q_val = model(s).gather(-1, a)
    q_target = r[..., None] + gamma * torch.max(target_model(s_next), -1)[0].unsqueeze(-1) * done_flag[..., None]
    q_target = q_target.transpose(1, 2)
    td_error = q_target.detach() - q_val
    
    huber_loss = cal_huberloss(td_error)
    element_wise_loss_quantile_huber_loss = torch.abs(
        model.toi_hats[..., None] - (td_error.detach() < 0).float()
        ) * huber_loss / kappa
    
    quantiles_loss = element_wise_loss_quantile_huber_loss.sum(1).mean()
    model.optimizer.zero_grad()
    quantiles_loss.backward()
    model.optimizer.step()


if __name__ == "__main__":
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
    model = QRDQN((state_size, output_size, learning_rate, device)).to(device)
    target_model = QRDQN((state_size, output_size, learning_rate, device)).to(device)
    target_model.load_state_dict(model.state_dict())
    replaybuffer = Replaybuffer((memory_len, device))

    for i in range(epoch_num):
        s = env.reset()
        score = 0.
        epsilon = max(0.01, epsilon * 0.999)
        for j in range(max_steps):
            # env.render()
            a = model.choose_action(s, epsilon)
            s_next, reward, done, info = env.step(a)
            done_flag = 0.0 if done else 1.0
            replaybuffer.save_memory((s, a, reward, s_next, done_flag))
            score += reward
            s = s_next
            if len(replaybuffer.buffer) >= train_len:
                train_flag = True
                train(model, target_model, replaybuffer, gamma, batch_size)
            if done:
                break
        # 更新目标网络
        if (i + 1) % update_target_interval == 0 and i > 0:
            target_model.load_state_dict(model.state_dict())
        print("{} epoch score: {}  training: {}  epsilon:{:.3}".format(i+1, score, train_flag, epsilon))
    env.close()


