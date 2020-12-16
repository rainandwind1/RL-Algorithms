import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import gym
import collections

class ReplayBuffer():
    def __init__(self, args):
        self.mem_size, self.device = args
        self.buffer = collections.deque(maxlen = self.mem_size)


    def save_trans(self, trans):
        self.buffer.append(trans)

    def sample_trans(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_ls = [([] for _ in range(5))]
        for trans in sample_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_ls.append([done_flag])
        return torch.FloatTensor(s_ls).to(self.device),\
                torch.LongTensor(a_ls).to(self.device),\
                torch.FloatTensor(r_ls).to(self.device),\
                torch.FloatTensor(s_next_ls).to(self.device),\
                torch.FloatTensor(done_ls).to(self.device)

class OPAC(nn.Module):
    def __init__(self, args):
        super(OPAC, self).__init__()
        self.input_size, self.output_size, self.sigma, self.alpha, self.lr, self.device = args
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_ls = nn.ModuleList([self.critic for _ in range(3)])
        self.optimizer_critic = optim.Adam(self.critic_ls.parameters(), lr = self.lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = self.lr)


    def choose_action(self, inputs):
        inputs = torch.FloatTensor(inputs).to(self.device)
        policy_op = self.actor(inputs)
        action = policy_op.item() + random.gauss(0, self.sigma)
        return action


    def train(self, 
    target_model, 
    replay_buffer,
    way = 'mean', 
    target_update = False,
    c = 0.5,   # hyperparam 
    batch_size = 32, 
    gamma = 0.98, 
    toi = 0.99, # hyperparam
    a_low = 0.1, # hyperparam
    a_high = 0.8, # hyperparam
    H0 = 1.2 # hyperparam
    ):
        epsilon = random.gauss(0, self.sigma)
        s, a, r, s_next, done = replay_buffer.sample_batch(batch_size)
        a_next = torch.clamp(target_model.actor(s_next) + torch.clamp(torch.ones_like(s_next)*epsilon, -c, c) , a_low, a_high)
        q_target_ls, q_ls = [], []
        for i in range(len(self.critic_ls)):
            q_target_ls.append(r + gamma * target_model.critic_ls[i](s_next) * done)
            q_ls.append(self.critic_ls[i](s))
        if way == 'mean':
            q_target_total = torch.cat(q_target_ls, 0)
            q_total = torch.cat(q_ls, 0)
            q_target = q_target_total.mean(-1) - self.alpha * torch.log(a_next)
            q_val = q_total.mean(-1)
        else:
            pass
        critic_loss = ((q_val - q_target)**2).mean()
        optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph = True)
        optimizer_critic.step()


        if target_update:
            # policy update
            q1_val = self.critic_ls[0](self.actor(s))
            actor_loss = q1_val.mean() - self.alpha * torch.log(self.actor(s)).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # target update
            for target_param, online_param in zip(model.parameters(), target_model.parameters()):
                target_param = toi * target_param + (1 - toi) * online_param
            
            # entropy coef update   lambda = 1e-4
            self.alpha += 1e-4 * (torch.log(self.actor(s)).mean() + H0)




if __name__ == "__main__":
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    SIGMA = 0.5
    LEARNING_RATE = 1e-3
    POLICY_DELAY = 2        # target / policy net(actor) update delayed after online net / critic net  
    H0 = 0.8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = OPAC(args = (INPUT_SIZE, OUTPUT_SIZE, SIGMA, LEARNING_RATE, DEVICE))
    target_model = OPAC(args = (INPUT_SIZE, OUTPUT_SIZE, SIGMA, LEARNING_RATE, DEVICE))
    target_model.load_state_dict(model.state_dict())
