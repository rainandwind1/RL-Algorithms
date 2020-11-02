import torch
import random
import gym
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import collections

class ReplayBuffer():
    def __init__(self, args):
        self.mem_size, self.device = args
        self.buffer = collections.deque(maxlen = self.mem_size)

    def save_trans(self, trans):
        self.buffer.append(trans)

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
        return torch.tensor(s_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(a_ls,dtype=torch.int64).to(self.device),\
            torch.tensor(r_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(s_next_ls,dtype=torch.float32).to(self.device),\
            torch.tensor(done_mask_ls,dtype=torch.float32).to(self.device)



class NDQFN(nn.Module):
    def __init__(self, args, Embedding_d = 32):
        super(NDQFN, self).__init__()
        self.input_size, self.output_size, self.lr, self.device, self.N, self.predef_p = args
        self.persi_net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, Embedding_d)
        )

        self.phi_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, Embedding_d)
        )

        self.baseline_f = nn.Sequential(
            nn.Linear(Embedding_d, 64),
            nn.Linear(64, self.output_size),
            nn.Sigmoid()
        )

        self.incremental_g = nn.Sequential(
            nn.Linear(2*Embedding_d, 64),
            nn.Linear(64, self.output_size),
            nn.ReLU()
        )

        self.optimizer = optim.Adam(self.parameters(), self.lr)
    

    def delta_w(self, j, inputs, predef_p, persi_net_op):
        if j == 0:
            return self.baseline_f(persi_net_op)
        if len(inputs.shape) == 1:
            delta_phi = self.phi_net(predef_p[j]) - self.phi_net(predef_p[j - 1])
        else:
            delta_phi = torch.cat([(self.phi_net(predef_p[j]) - self.phi_net(predef_p[j - 1])).unsqueeze(0) for i in range(inputs.shape[0])], 0)
        return self.incremental_g(torch.cat([persi_net_op*self.phi_net(predef_p[j]), delta_phi], -1))


    def G_avg(self, toi, inputs, predef_p, choose_action = False):
        res = []
        persi_net_op = self.persi_net(inputs)
        delta_0_omga = self.baseline_f(persi_net_op)
        toi_sum_ls = [delta_0_omga]
        toi_sum = 0
        delta_ls = []
        for i in range(1, predef_p.shape[0] - 1):
            delta_i = self.delta_w(i, inputs, predef_p, persi_net_op)
            toi_sum += delta_i
            delta_ls.append(delta_i)
            toi_sum_ls.append(toi_sum)
        delta_ls.append(self.delta_w(self.N, inputs, predef_p, persi_net_op))
        for i in range(self.N):  # N 个 toi 遍历
            ans = 0
            if choose_action:
                idx = i
            else:
                for ip in range(predef_p.shape[0] - 1):
                    if predef_p[ip] <= toi[i] < predef_p[ip + 1]:
                        idx = ip
                        break 
            ans += toi_sum_ls[idx]
            ans += (toi[i] - predef_p[idx]) / (predef_p[idx + 1] - predef_p[idx]) * delta_ls[idx]
            res.append(ans.unsqueeze(0))
        res = torch.cat(res, 0)
        return res
    

    def choose_action(self, inputs):   
        inputs = torch.FloatTensor(inputs).to(self.device)
        quantile_pred = self.G_avg(self.predef_p, inputs, self.predef_p, choose_action = True)
        delta_p = (self.predef_p[1:self.N] - self.predef_p[0:self.N-1]) / 2
        Q_val = (delta_p * (quantile_pred[1:self.N, :] + quantile_pred[0:self.N-1, :])).mean(0)
        return int(torch.argmax(Q_val).item())


    def train(self, target_model, replaybuffer, batch_size, gamma = 0.98): 
        s, a, r, s_next, done = replaybuffer.sample_batch(batch_size)
        toi_1 = torch.FloatTensor([[np.random.uniform()] for i in range(self.N)]).to(self.device)
        toi_2 = torch.FloatTensor([[np.random.uniform()] for i in range(self.N)]).to(self.device)
        
        z_val = self.G_avg(toi_1, s, self.predef_p).permute(1, 2, 0)
        z_target = target_model.G_avg(toi_2, s_next, self.predef_p).permute(1, 2, 0)
        
        z_val = torch.stack([z_val[i].index_select(0, a[i]) for i in range(batch_size)]).squeeze(1).unsqueeze(-1)
        a_next = z_target.mean(-1).argmax(-1)
        z_target = torch.stack([z_target[i].index_select(0, a_next[i]) for i in range(batch_size)]).squeeze(1)
        z_target = (r + gamma * z_target * done).unsqueeze(-2)

        delta_ij = z_target.detach() - z_val
        toi_1 = toi_1.unsqueeze(0)
        weight = torch.abs(toi_1 - delta_ij.le(0.).float())
        loss = F.smooth_l1_loss(z_val, z_target.detach())
        loss = torch.mean(weight * loss, 1).mean(1)
        loss = torch.mean(torch.ones_like(r).unsqueeze(-1) * loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




if __name__ == "__main__":
    # Hyperparameter
    N = 30
    EPSILON = 1e-5
    LEARNING_RATE = 1e-3
    MEM_SIZE = 30000
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_EPOCH = 100000
    

    predef_p = [np.random.uniform() for i in range(N + 1)]
    predef_p[0], predef_p[-1] = EPSILON, 1 - EPSILON
    predef_p.sort()
    predef_p = torch.FloatTensor(predef_p).to(DEVICE).unsqueeze(-1)


    # # self.input_size, self.output_size, self.lr, self.device, self.N, self.predef_p = args
    # # self.mem_size, self.device = args
    model = NDQFN(args = (4, 2, LEARNING_RATE, DEVICE, N, predef_p)).to(DEVICE)
    target_model = NDQFN(args = (4, 2, LEARNING_RATE, DEVICE, N, predef_p)).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    replaybuffer = ReplayBuffer(args = (MEM_SIZE, DEVICE))

    done = False
    train_flag = False
    total_step = 0
    env = gym.make("CartPole-v1")
    for epo_i in range(MAX_EPOCH):
        s = env.reset()
        done = False
        score = 0
        while not done:
            total_step += 1
            a = model.choose_action(s)
            # a = random.sample(range(2),1)[0]
            s_next, r, done, info = env.step(a)

            score += r
            replaybuffer.save_trans((s, a, r, s_next, done))
            s = s_next
            if total_step > 300:
                train_flag = True
                # train(self, target_model, replaybuffer, batch_size):
                model.train(target_model, replaybuffer, BATCH_SIZE)

            if done:
                print("Epoch:{} score:{} training:{}".format(epo_i, score, train_flag))

        if epo_i % 30 == 0 and epo_i > 0:
            target_model.load_state_dict(model.state_dict())






    # toi_1 = torch.FloatTensor([[np.random.uniform()] for i in range(N)]).to(DEVICE)
    # for i in range(100):   # test for choose action
    #     state = torch.randn(1, 4)
    #     model.G_avg(toi_1, state, predef_p)
    #     print(i) 
