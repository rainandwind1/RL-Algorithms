import torch
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import gym
import math



class ReplayBuffer():
    def __init__(self, args):
        self.name, self.mem_size, self.device = args
        self.replaybuffer = collections.deque(maxlen = self.mem_size)
    
    def save_trans(self, trans):
        self.replaybuffer.append(trans)
    
    # 待修改
    def sample_batch(self, batch_size):
        trans_batch = random.sample(self.replaybuffer, batch_size)
        if self.name == "option_rb":
            option_id_ls, a_seq_ls, s_seq_ls = ([] for _ in range(3))
            for trans in trans_batch:
                option_id, a_avg, s_avg = trans
                option_id_ls.append(option_id)
                a_seq_ls.append(a_avg)
                s_seq_ls.append(s_avg)
            return option_id_ls,\
                    a_seq_ls,\
                    s_seq_ls

        elif self.name == "sche_rb":
            s0_ls, reward_acc_ls, s_next0_ls, option_id_ls = ([] for _ in range(4))
            for trans in trans_batch:
                s0, option_id, s_next0, reward_acc = trans
                s0_ls.append(s0)
                reward_acc_ls.append([reward_acc])
                s_next0_ls.append(s_next0)
                option_id_ls.append(option_id)
            return torch.FloatTensor(s0_ls).to(self.device),\ 
                    option_id_ls,\
                    torch.FloatTensor(s_next0_ls).to(self.device),\
                    torch.FloatTensor(reward_acc_ls).to(self.device)
                    



class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.name, self.input_size, self.output_size, self.mem_size, self.device = args
        self.net = nn.Sequentional(
            nn.Linear(self.input_size, 128),
            nn.Relu(),
            nn.Linear(128, 64)
            nn.Relu(),
            nn.Linear(64, self.output_size)
        )
        

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs).to(self.device)
        policy_op = self.net(inputs)
        return F.softmax(policy_op)


class HIDIO(nn.Module):
    def __init__(self, args):
        super(HIDIO, self).__init__()
        self.input_size, self.output_size, self.option_num, self.mem_size, self.lr, self.device = args
        self.option_phi = Policy(args = ("option_phi", self.input_size, self.output_size, self.mem_size, self.device))
        self.policy_sche = Policy(args = ("scheduler_policy", self.input_size + 1, self.option_num, self.mem_size, self.device))
        self.optimizer_sche = optim.Adam(self.policy_sche.parameter(), self.lr)
        self.optimizer_option_phi = optim.Adam([{'params': self.option_phi.parameter()},{'params': self.persi_net.parameter()}], self.lr)
        self.sche_replay_buffer = ReplayBuffer(args = ("sche_rb", self.mem_size, self.device))
        self.option_replay_buffer = ReplayBuffer(args = ("option_rb", self.mem_size, self.device))
        self.persi_net = nn.Sequentional(
            nn.Linear(self.input_size + self.output_size, 128),
            nn.Relu(),
            nn.Linear(128, 64),
            nn.Relu(),
            nn.Linear(64, 1)
        )

    def choose_option(self, inputs):
        inputs = torch.FloatTensor(inputs).to(self.device)
        return int(torch.argmax(F.softmax(self.policy_sche(inputs))).item())

    def choose_action(self, inputs, option_id):
        action_prob = self.option_ls[option_id](inputs)
        return int(torch.argmax(action_prob).item())

    def cal_sche_target(self, inputs):


    def cal_option_target(self, inputs):


    def train(self, gamma, train_time, batch_size):
        for m in range(train_time):
            # scheduler training
            s0, option_id, s_next0, r_acc  = self.sche_replay_buffer.sample_batch(batch_size)




            # worker training
            option_id, a_seq, s_seq = self.option_replay_buffer.sample_batch(batch_size)




if __name__ == "__main__":
    # Hyperparameters
    T = 300         # episode cycle interval   Epsisode length
    BATCH_SIZE = 64 # batch size
    K = 20          # option cycle interval
    M = 6           # batches per iteration
    LEARNING_RATE = 1e-3
    H = int(T/K)    # Pi policy cycle interval
    OPTION_NUM = 3
    GAMMA_SCHE = 0.98 # Scheduler gamma
    GAMMA = 0.98
    BETA_op = 0.98    # option beta
    BETA_R = 0.01
    MEM_SZIE = 20000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_EPISODES = 100000
    RENDER_FLAG = False
    
    train_flag = False
    env = gym.make("CartPole-v1")
    print("Obversation space: ", env.obversation_space)
    print("Action space: ", env.action_space)

    # Model initialize
    model = HIDIO(args = (4, 2, OPTION_NUM, MEM_SIZE, LEARNING_RATE, DEVICE))

    # Rollout
    for epi_i in range(MAX_EPISODES):
        s = env.reset()
        s_seq = [s]
        a_seq = []
        epi_step = 0
        done = False
        for option_h in range(T/K):
            s0 = s
            option_id = model.choose_option(s)
            r_acc = 0
            r_ls = []
            for i in range(K):
                if RENDER_FLAG:env.render()
                epi_step += 1
                a = model.choose_action(s, option_id)
                s, a, r, s_next, done = env.step(a)
                
                s_seq.append(s_next) 
                a_seq.append(a)
                
                s = s_next
                r_ls.append(r)
                # save memmory
                model.sche_replay_buffer.save_trans((option_id, a_seq, s_seq))
                if done:break
            for r in r_ls[::-1]:
                r_acc = 0.98 * r_acc + r # gamma_k 
            s_next0 = s_next
            model.option_replay_buffer.save_trans((s0, option_id, s_next0, r_acc))
            if done:break
        if epi_i > 10:
            train_flag = True
            model.train(GAMMA, M, BATCH_SIZE)
        print("Epoch:{}  reward:{}  training:{}".format(epi_i + 1, reward_acc, train_flag))
        



    

    





