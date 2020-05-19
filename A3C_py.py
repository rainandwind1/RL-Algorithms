import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
import random
import collections
import matplotlib.pyplot as plt
import gym
import threading
from torch.autograd import Variable


THREAD_NUM = 8
MAX_STEPS = 20
Total_step = 0
GAMMA= 0.95
Loss_list = []
Score_list = []
score_episode = [0. for _ in range(THREAD_NUM)]
Policy_coef = 0.3
Critic_coef = 0.6
LEARNING_RATE = 1e-3
epoch = 0
max_score = 0.


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, self.output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, inputs):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        op = self.net(inputs)
        actor = F.softmax(self.actor(op), dim = 0)
        critic = self.critic(op)
        return actor, critic


def make_envs(nums):
    envs = []
    for i in range(THREAD_NUM):
        envs.append(gym.make("CartPole-v1"))
        print("Initializing {} env ...".format(i))
    return envs



def runprocess(envs, model, thread_id, s_t, optimizer):
    global Total_step
    global score_episode
    global epoch
    global max_score
    r_ls = []
    critic_ls = []
    policy_ls = []
    trans_list = []

    step = 0
    while step < MAX_STEPS:

        policy, critic = model(s_t)

        action_choice = torch.multinomial(policy, 1)
        action_choice = int(action_choice[0])
        # print(action_choice)
        # envs[thread_id].render() # 刷新显示
        s_next, reward, done, info = envs[thread_id].step(action_choice)

        score_episode[thread_id] += reward
        policy_ls.append(policy[action_choice])
        critic_ls.append(critic)
        r_ls.append(reward)

        if done:
            done_flag = 0.
        else:
            r_ls[-1] = critic_ls[-1]
            done_flag = 1.
        trans_list.append((s_t, action_choice, reward, s_next, done_flag))

        s_t = s_next
        if done or step == MAX_STEPS:
            for i in reversed(range(len(r_ls) - 1)):
                r_ls[i] = r_ls[i] + GAMMA * r_ls[i + 1]
            if done:
                Score_list.append(score_episode[thread_id])
                max_score = max(max_score, score_episode[thread_id])
                print("Epoch: {} score: {} max_score: {}".format(epoch, score_episode[thread_id], max_score))
                epoch += 1
                score_episode[thread_id] = 0.
                s_t = envs[thread_id].reset()
        step += 1
        Total_step += 1
    train_net(model, optimizer, trans_list, r_ls, Loss_list)
    return s_t # 返回当前环境的停止状态， 下次执行当前线程时紧接着上次的状态执行


def train_net(model, optimizer, trans_list, return_list, loss_list):
    for num, trans in enumerate(trans_list):
        s, a, r, s_next, done_flag = trans
        policy, critic = model(s)
        # advantage = r + GAMMA*model(s_next)[1]*done_flag - critic
        #
        advantage = return_list[num] - critic
        loss_critic = advantage**2
        loss_policy = torch.log(policy[a])*advantage
        loss = -Policy_coef * loss_policy + Critic_coef * loss_critic

        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()



class actor_threads(threading.Thread):
    def __init__(self, envs, model, thread_id, s_t, optimizer):
        threading.Thread.__init__(self)
        self.envs = envs
        self.model = model
        self.thread_id = thread_id
        self.optimizer = optimizer
        self.init_state = s_t

    def run(self):
        threadLock.acquire()
        self.init_state = runprocess(self.envs, self.model, self.thread_id, self.init_state, self.optimizer)
        threadLock.release()



if __name__ == "__main__":
    # 创建线程环境
    envs = make_envs(THREAD_NUM)
    s_init_list  = []
    for i in range(THREAD_NUM):
        s_init_list.append(envs[i].reset())

    model = ActorCritic(4, 2)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    while True:
        thread = []
        threadLock = threading.Lock()
        # 创建线程
        for i in range(THREAD_NUM):
            thread.append(actor_threads(envs, model, i, s_init_list[i], optimizer))

        # 线程执行
        for i in range(THREAD_NUM):
            thread[i].start()

        # 等待所有线程退出
        for i in range(THREAD_NUM):
            thread[i].join()

        # 刷新状态
        for i in range(THREAD_NUM):
            s_init_list[i] = thread[i].init_state