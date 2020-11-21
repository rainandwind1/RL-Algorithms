import torch
import random
import numpy as np
import gym
from torch import nn, optim
import torch.nn.functional as F
import collections



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


class CrossDQN(nn.Module):
    def __init__(self, args):
        super(CrossDQN, self).__init__()
        self.input_size, self.output_size, self.lr, self.K, self.device = args
        self.share_net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.q_net_ls = nn.ModuleList([nn.Linear(64, self.output_size) for i in range(self.K)])
        self.optimizer = [optim.Adam( [{'params': self.share_net.parameters(), 'params':self.q_net_ls[i].parameters()}], lr = self.lr) for i in range(self.K)]


    def cal_qval(self, inputs, idx):
        embedding_op = self.share_net(inputs)
        q_val = self.q_net_ls[idx](embedding_op)
        return q_val

    def choose_action(self, inputs, epsilon):
        inputs = torch.FloatTensor(inputs).to(self.device)
        embedding_op = self.share_net(inputs)
        candidate_ls = []
        for i in range(self.K):
            q_val = self.q_net_ls[i](embedding_op)
            if np.random.rand() > epsilon:
                candidate_ls.append(torch.argmax(q_val, -1).item())
            else:
                candidate_ls.append(random.sample(range(self.output_size), 1)[0])
        action = candidate_ls[np.argmax(np.bincount(candidate_ls))]
        return action


    def train(self, replay_buffer, batch_size, gamma = 0.98):
        s, a, r, s_next, done = replay_buffer.sample_memory(BATCH_SIZE)
        for idx in range(self.K):
            q_val = self.cal_qval(s, idx)
            q_val = q_val.gather(-1, a)
            next_qval = self.cal_qval(s_next, idx)
            a_next = torch.cuda.LongTensor(torch.argmax(next_qval, -1).unsqueeze(-1))
            target_idx = random.sample(range(self.K), 1)[0]
            q_target = self.cal_qval(s_next, target_idx)
            q_target = q_target.gather(-1, a_next)
            td_error = ((r + gamma * q_target.detach() * done - q_val)**2 / (self.K ** 2)).mean()
            self.optimizer[idx].zero_grad()
            td_error.backward()
            self.optimizer[idx].step()



if __name__ == "__main__":
    K = 5
    MAX_EPOCH = 100000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CrossDQN(args = (4, 2, LEARNING_RATE, K, DEVICE)).to(DEVICE)
    replay_buffer = Replaybuffer(args = (30000, DEVICE))

    env = gym.make('CartPole-v1')
    train_flag = False
    epsilon = 0.9
    for epo_i in range(MAX_EPOCH):
        epsilon = max(0.01, epsilon * 0.999)
        s = env.reset()
        done = False
        score = 0
        while not done:
            action = model.choose_action(s, epsilon)
            s_next, r, done, info = env.step(action)

            score += r
            replay_buffer.save_memory((s, action, r, s_next, done))

            s = s_next
            if len(replay_buffer.buffer) > 2000:
                train_flag = True
                model.train(replay_buffer, BATCH_SIZE)
            if done:
                print("Epoch:{}  score:{}   training:{}  epsilon:{:.3}".format(epo_i + 1, score, train_flag, epsilon))








