import torch
import gym
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import collections
import random


class DQN(nn.Module):
    def __init__(self, input_size, output_size, lr):
        super(DQN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, inputs):
        return self.net(inputs)

    def choose_action(self, inputs, epsilon):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        output = self.net(inputs)
        coin = np.random.rand()
        if coin > epsilon:
            return int(torch.argmax(output))
        else:
            return np.random.randint(0, self.output_size)



class Memory():
    def __init__(self, mem_lem):
        self.memory = collections.deque(maxlen = mem_lem)

    def save_trans(self, trans):
        self.memory.append(trans)


    def sample_batch(self, batch_size):
        trans_batch = random.sample(self.memory, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls, dtype=torch.float32),\
               torch.tensor(a_ls, dtype=torch.int64),\
               torch.tensor(r_ls, dtype=torch.float32),\
               torch.tensor(s_next_ls, dtype=torch.float32),\
               torch.tensor(done_flag_ls, dtype=torch.float32)


def train(mode1, model_target, memory, gamma, batch_size, state):
    s, a, r, s_next, done_flag = memory.sample_batch(batch_size)
    q_value = mode1(s)
    q_value = torch.gather(q_value, 1, a)

    q_target = model_target(s_next).detach()
    if state == 'main':
        q_target = r + gamma * torch.max(q_target, dim=1, keepdim=True)[0] * done_flag
    elif state == 'adver':
        q_target = -r + gamma * torch.max(q_target, dim=1, keepdim=True)[0] * done_flag
    # loss = (q_target - q_value)**2
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_value, q_target)

    mode1.optimizer.zero_grad()
    loss.backward()
    mode1.optimizer.step()



if __name__ == "__main__":
    epsilon = 0.3
    BATCH_SIZE = 32
    GAMMA = 0.98
    MAX_EPOCH = 10000
    COVER_INTERVAL = 20
    MEM_LEN = 30000
    LEARNING_RATE = 1e-3
    RENDER = True

    state = 'main'
    count_main = 25
    count_adver = 25

    main_agent = DQN(8,4, LEARNING_RATE)
    main_target = DQN(8,4, LEARNING_RATE)
    main_target.load_state_dict(main_agent.state_dict())

    adver_agent = DQN(8,4, LEARNING_RATE)
    adver_target = DQN(8,4, LEARNING_RATE)
    adver_target.load_state_dict(adver_agent.state_dict())

    memory = Memory(MEM_LEN)
    env = gym.make('LunarLander-v2')
    total_step = 0
    train_flag = False
    for epo_i in range(MAX_EPOCH):
        done = False
        epsilon = max(epsilon*0.99, 0.01)
        s = env.reset()
        score = 0
        # 主代理
        if state == 'main':
            while not done:
                action = main_agent.choose_action(s, epsilon)
                # print(action)
                if RENDER:
                    env.render() 
                s_next, r, done, info = env.step(action)
                done_flag = 0 if done else 1
                memory.save_trans((s, action, r, s_next, done_flag))

                total_step += 1
                score += r
                if total_step > 1000:
                    train_flag = True
                    train(main_agent, main_target, memory, GAMMA, BATCH_SIZE, state)
                if done:
                    print("epo: {}  score: {}  training: {}  agent state: {}".format(epo_i, score, train_flag, state))
                s = s_next

            count_main -= 1
            if count_main <= 0:
                state = 'adver'
                count_main = 25

        # 对抗代理
        elif state == 'adver':
            while not done:
                action = adver_agent.choose_action(s, epsilon)
                if RENDER:
                    env.render() 
                s_next, r, done, info = env.step(action)
                done_flag = 0 if done else 1
                memory.save_trans((s, action, r, s_next, done_flag))

                total_step += 1
                score += r
                if total_step > 2000:
                    train_flag = True
                    train(adver_agent, adver_target, memory, GAMMA, BATCH_SIZE, state)
                if done:
                    print("epo: {}  score: {}  training: {}  agent state: {}".format(epo_i, score, train_flag, state))
                s = s_next


            count_adver -= 1
            if count_adver <= 0:
                state = 'main'
                count_adver = 25

        if epo_i % COVER_INTERVAL == 0 and epo_i != 0:
            main_target.load_state_dict(main_agent.state_dict())
            adver_target.load_state_dict(adver_target.state_dict())


