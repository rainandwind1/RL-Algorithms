'''
code copy movan
'''
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
Total_epoch = 0

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.policy = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.distribution = torch.distributions.Categorical
        self.mem = []
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, inputs):
        policy = self.policy(inputs)
        critic = self.critic(inputs)
        action_prob = F.softmax(policy, 0).data
        return action_prob, critic


    def choose_action(self, inputs):
        self.eval()
        inputs = torch.tensor(inputs, dtype = torch.float32)
        action_prob, val = self(inputs)
        action = self.distribution(action_prob)
        return action.sample().numpy()

    def save_trans(self, trans):
        self.mem.append(trans)

    def train_net(self, gamma, ac_net):
        for trans in self.mem:
            s, a, r, s_next, done_flag = trans
            s = torch.tensor(s, dtype = torch.float32)
            s_next = torch.tensor(s_next, dtype = torch.float32)
            q_target = r + gamma * self.critic(s_next) * done_flag
            q_val = self.critic(s)
            td_error = q_target - q_val
            action_p = F.softmax(self.policy(s), 0)[a]
            loss_policy = -torch.log(action_p) * td_error.detach()
            loss_critic = td_error ** 2
            loss = loss_critic + loss_policy
            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            self.optimizer.step()
        self.mem = []
        ac_net.load_state_dict(self.state_dict())

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, gamma):
        super(Worker, self).__init__()
        global Total_epoch
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt =  gnet, opt
        self.lnet = Net(N_S, N_A)
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.env = gym.make('CartPole-v0').unwrapped
        self.gamma = gamma

    def run(self):
        global Total_epoch
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            score = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(s)
                s_next, r, done, info = self.env.step(a)
                done_flag = 0. if done else 1.
                score += r
                self.gnet.save_trans((s, a, r, s_next, done_flag))

                s = s_next
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.gnet.train_net(gamma=self.gamma, ac_net=self.lnet)
                    if done:
                        self.res_queue.put(score)
                        Total_epoch += 1
                        print("Epoch:{}   score:{}".format(Total_epoch, score))
                        break
            self.res_queue.put(None)

if __name__ == "__main__":
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet,None, global_ep, global_ep_r, res_queue, i, gamma=GAMMA) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


