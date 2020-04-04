import gym
import torch
from torch import nn,optim
import numpy as np
from DQN_Prioritized_Replay_py import DQN, train, plot_curse
import copy

env = gym.make("CartPole-v1")
# env = gym.make("Acrobot-v1")
# env = gym.make("Breakout-ram-v0")
obversation = env.reset()

print("Obversation space:",env.observation_space)
print("Action space:",env.action_space)



# 超参数设置
gamma = 0.90
learning_rate = 0.01
output_size = 2
state_size = 4
memory_len = 2000
#alpha = 0.6   ???


epoch_num = 2000   # 回合数
max_steps = 400   # 最大步数
update_target_interval = 50 # 目标网络更新间隔
batch_size = 64

# 初始化
Q_value = DQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
Q_target =  DQN(input_size = state_size,output_size=output_size,memory_len = memory_len)


score_list = []
loss_list = []
train_step = 0
huber = nn.SmoothL1Loss()
optimizer = optim.Adam(Q_value.parameters(), lr = learning_rate)

for i in range(epoch_num):
    epsilon = max(0.01,0.16-0.01*(i)/200)
    s = env.reset()
    score = 0
    for j in range(max_steps):
        env.render()
        a = Q_value.sample_action(s,epsilon=epsilon)
        s_next,reward,done,info = env.step(a)
        done_flag = 0.0 if done else 1.0
        Q_value.save_memory(Q_target,(s,a,reward/100,s_next,done_flag),huber,gamma)
        score += reward
        s = s_next
        if done:
            break
    score_list.append(score)
    if len(Q_value.memory_list) >= memory_len:
        train_step += 1
        if train_step == 1:
            print("train begin!")
        train(Q_value,Q_target,batch_size,gamma,learning_rate,loss_list,Replay_time=20)
    # 更新目标网络
    if (i+1) % update_target_interval == 0 and i > 0:
        Q_target.load_state_dict(Q_value.state_dict())
        print("Target net load weight ! %d epochs score: %d \n"%(i+1,score))



plot_curse(score_list,loss_list)
env.close()


