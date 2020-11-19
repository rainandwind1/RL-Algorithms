import torch
import gym
from torch import nn, optim
import random
import collections
import numpy as np




class ReplayBuffer():
    def __init__(self, args):
        self.mem_size, self.device = args
        self.buffer = collections.deque(maxlen=self.mem_size)

    def save_trans(self, trans):
        self.buffer.append(trans)
    
    def sample_batch(self, batch_size):
        s_ls, a_ls, r_ls, F_ls, s_next_ls, done_ls = ([] for _ in range(6)) # Fi => feature vector
        trans_batch = random.sample(self.buffer, batch_size)
        for trans in trans_batch:
            s, a, r, F, s_next, done = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            F_ls.append(s)
            s_next_ls.append(s_next)
            done_ls.append([done])
        return torch.FloatTensor(s_ls).to(self.device),\
                torch.LongTensor(a_ls).to(self.device),\
                torch.FloatTensor(r_ls).to(self.device),\
                torch.FloatTensor(F_ls).to(self.device),\
                torch.FloatTensor(s_next_ls).to(self.device),\
                torch.FloatTensor(done_ls).to(self.device)


class Esp_dqn(nn.Module):
    def __init__(self, args):
        super(Esp_dqn, self).__init__()
        self.input_size, self.output_size, self.lr, self.device = args
        self.n_dim = 4
        self.qf_net = nn.Sequential(
            nn.Linear(self.input_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_dim),       # embedding size:32
            nn.ReLU()
        )
        self.c_net = nn.Sequential(
            nn.Linear(self.n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer_qf = optim.Adam(self.qf_net.parameters(), lr = self.lr)
        self.optimizer_c = optim.Adam(self.c_net.parameters(), lr = self.lr)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def cal_fop_q_val(self, s, a = None):
        if a is None:
            f_op_ls = []
            q_val_ls = []
            for i in range(self.output_size):
                a_i = torch.cat([torch.FloatTensor([i]).to(self.device).unsqueeze(0) for i in range(s.shape[0])], 0)
                inputs = torch.cat([s, a_i], -1)
                f_op_ = self.qf_net(inputs)
                q_val_ = self.c_net(f_op_)
                q_val_ls.append(q_val_)
                f_op_ls.append(f_op_.unsqueeze(1))
            f_op = torch.cat(f_op_ls, 1)
            q_val = torch.cat(q_val_ls, -1)
            max_qval, max_index = torch.max(q_val, -1)
            max_qval_end = max_qval.unsqueeze(-1)
            max_index = torch.cuda.LongTensor(max_index.unsqueeze(-1))
            max_index_end = torch.cat([max_index for _ in range(self.n_dim)], -1).unsqueeze(-2)
            f_op_end = torch.gather(f_op, -2, max_index_end).squeeze(-2)
            return f_op_end, max_qval_end
        else:
            inputs = torch.cat([s, a], -1)
            f_op = self.qf_net(inputs)
            q_val = self.c_net(f_op)
            return f_op, q_val


    def choose_action(self, inputs, epsilon):
        inputs = torch.FloatTensor(inputs).to(self.device)
        q_val_ls = []
        f_op_ls = []
        for i in range(self.output_size):
            inputs_i = torch.cat([inputs, torch.FloatTensor([i]).to(self.device)], -1)
            f_op = self.qf_net(inputs_i)
            q_val = self.c_net(f_op)
            f_op_ls.append(f_op)
            q_val_ls.append(q_val)
        coin = np.random.rand()
        if coin > epsilon:
            action = q_val_ls.index(max(q_val_ls))
        else:
            action = random.sample(range(self.output_size), 1)[0]
        return action, f_op_ls[action]
        



    def train(self, target_model, replay_buffer, batch_size = 32, gamma = 0.98, beta = 0.98):
        s, a, r, F_ip, s_next, done = replay_buffer.sample_batch(batch_size)
        qf_op, q_val = self.cal_fop_q_val(s, a)
        tqf_op, tq_val_raw = target_model.cal_fop_q_val(s_next)       

        tqf_val = F_ip + gamma * tqf_op * done  # torch.rand_like(tqf_op).to(self.device)
        tq_val = r + gamma * tq_val_raw * done
        
        loss_c =  ((tq_val - q_val) ** 2).mean()
        loss_qf = ((tqf_val - qf_op) ** 2 ).mean()


        self.optimizer_c.zero_grad()
        loss_c.backward(retain_graph = True)
        self.optimizer_c.step()

        
        self.optimizer_qf.zero_grad()
        loss_qf.backward()
        self.optimizer_qf.step()
        




if __name__ == "__main__":
    K_HORIZON = 30
    MAX_EPOCH = 100000
    MEM_SIZE = 30000
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Esp_dqn(args = (4, 2, 1e-3, DEVICE)).to(DEVICE)
    target_model = Esp_dqn(args = (4, 2, 1e-3, DEVICE)).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    replaybuffer = ReplayBuffer(args = (MEM_SIZE, DEVICE))

    env = gym.make("CartPole-v1")
    train_flag = False
    epsilon = 0.8
    for epo_i in range(MAX_EPOCH):
        s = env.reset()
        done = False
        score = 0
        epsilon = max(0.01, epsilon * 0.999)
        while not done:
            action, F_op = model.choose_action(s, epsilon)
            s_next, r, done, info = env.step(action)
            
            score += r
            replaybuffer.save_trans((s, action, r, F_op, s_next, done))
            
            s = s_next
            if len(replaybuffer.buffer) > 1000: #10*BATCH_SIZE:
                train_flag = True
                model.train(target_model, replaybuffer)
            if done:
                print("Epoch:{} score:{} training:{} epsilon:{:.3f}".format(epo_i+1, score, train_flag, epsilon))
        if (epo_i+1) % K_HORIZON == 0:
            target_model.load_state_dict(model.state_dict())



# if __name__ == "__main__":
    
#     MAX_EPOCH = 10000

#     env = gym.make('LunarLander-v2')
#     env.seed(0)

#     print('State shape:', env.observation_space.shape)
#     print('Number of actions:', env.action_space.n)

#     state = env.reset()
#     for epo_i in range(MAX_EPOCH):
#         done = False
#         while not done:
#             action = random.sample(range(env.action_space.n), 1)[0]
#             env.render()
#             state, reward, done, info = env.step(action)
#             if done:
#                 state = env.reset()
#     env.close()