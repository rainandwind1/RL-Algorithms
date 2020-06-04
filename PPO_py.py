import gym
import torch
from torch import nn, optim
import torch.nn.functional as F
from  torch.distributions import Categorical



class PPO(nn.Module):
    def __init__(self, input_size, out_size):
        super(PPO, self).__init__()
        self.input_size = input_size
        self.output_size = out_size
        self.mem = []
        # net
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, self.output_size)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)

    def get_policy(self, inputs, dim):
        fc  = self.net(inputs)
        policy = self.policy(fc)
        policy = F.softmax(policy, dim = dim)
        return policy

    def get_critic(self, inputs):
        fc = self.net(inputs)
        critic = self.critic(fc)
        return critic

    def save_trans(self, trans):
        self.mem.append(trans)

    def package_trans(self):
        s_ls, a_ls, r_ls, s_next_ls, a_prob_ls, done_flag_ls = [], [], [], [], [], []
        for trans in self.mem:
            s, a, r, s_next, a_prob, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            a_prob_ls.append([a_prob])
            done_flag_ls.append([done_flag])
        s, a, r, s_next, a_prob, done_flag = torch.tensor(s_ls, dtype = torch.float32),\
                                                torch.tensor(a_ls, dtype = torch.int64),\
                                                torch.tensor(r_ls, dtype = torch.float32),\
                                                torch.tensor(s_next_ls, dtype = torch.float32),\
                                                torch.tensor(a_prob_ls, dtype = torch.float32),\
                                                torch.tensor(done_flag_ls, dtype = torch.float32)
        self.mem = []
        return s, a, r, s_next, a_prob, done_flag



def train(model, loss_fn, loss_list, score_list):
    s, a, r, s_next, a_prob, done_flag = model.package_trans()
    for i in range(K_EPOCH):
        td_target = r + GAMMA*model.get_critic(s_next)*done_flag
        td_error = td_target - model.get_critic(s)
        td_error = td_error.detach().numpy()

        advantage_ls = []
        advantage = 0.
        for error in td_error[::-1]:
            advantage = GAMMA * LAMBDA * advantage + error[0]
            advantage_ls.append([advantage])
        advantage_ls.reverse()
        advantage = torch.tensor(advantage_ls, dtype = torch.float32)

        policy = model.get_policy(s, 1)
        policy = policy.gather(1, a)
        ratio = torch.exp(torch.log(policy) - torch.log(a_prob)) # 重要性采样比率？

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP)*advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.get_critic(s), td_target.detach())

        model.optimizer.zero_grad()
        loss.mean().backward()
        model.optimizer.step()


# Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.98
LAMBDA = 0.95
EPS_CLIP = 0.1
K_EPOCH = 3
T_HORIZON = 20
MAX_EPOCH = 100000

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = PPO(4, 2)
    score = 0.
    done = False

    for epo_i in range(MAX_EPOCH):
        obs = env.reset()
        done = False
        while not done:
            for step in range(T_HORIZON):
                env.render()
                a_prob = model.get_policy(torch.from_numpy(obs).float(), 0)
                m = Categorical(a_prob)
                a = m.sample().item()
                obs_next, r, done, info = env.step(a)

                done_flag = 1.0 if not done else 0
                model.save_trans((obs, a, r, obs_next, a_prob[a].item(), done_flag))

                obs = obs_next
                score += r

                if done:
                    print("Epoch: {}  score: {}".format(epo_i, score))
                    score = 0.
                    break
            train(model, loss_fn=None, loss_list=None, score_list=None)
    env.close()