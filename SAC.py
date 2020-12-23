import torch
import gym
from torch import nn, optim
import gym
import random
import collections
import numpy as np
from torch.distributions import Normal


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low[0]
        high = self.action_space.high[0]

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return [action]

    def reverse_action(self, action):
        low = self.action_space.low[0]
        high = self.action_space.high[0]

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return [action]

class ReplayBuffer():
    def __init__(self, args):
        self.mem_size, self.device = args
        self.buffer = collections.deque(maxlen = self.mem_size)

    def save_trans(self, trans):
        self.buffer.append(trans)

    def sample_batch(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_ls = ([] for _ in range(5))
        for trans in sample_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_ls.append([done_flag])
        return torch.FloatTensor(s_ls).to(self.device),\
                torch.FloatTensor(a_ls).to(self.device),\
                torch.FloatTensor(r_ls).to(self.device),\
                torch.FloatTensor(s_next_ls).to(self.device),\
                torch.FloatTensor(done_ls).to(self.device)

class SAC(nn.Module):
    def __init__(self, args):
        super(SAC, self).__init__()
        self.input_size, self.output_size, self.lr, self.device = args
        self.min_sigma, self.max_sigma = -20, 2
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.embedding_mu = nn.Linear(32, 1)
        self.embedding_sigma = nn.Linear(32, 1)
        self.critic = nn.Sequential(
            nn.Linear(self.input_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_ls = nn.ModuleList([self.critic for _ in range(2)])
        self.optimizer_critic = optim.Adam(self.critic_ls.parameters(), lr = self.lr)
        self.optimizer_actor = optim.Adam([{'params':self.actor.parameters()}, {'params':self.embedding_mu.parameters()}, {'params':self.embedding_sigma.parameters()}], lr = self.lr)

    def choose_action(self, inputs):
        inputs = torch.FloatTensor(inputs).to(self.device)
        mu, sigma = self.embedding_mu(self.actor(inputs)), self.embedding_sigma(self.actor(inputs))
        sigma = torch.clamp(sigma, self.min_sigma, self.max_sigma)
        sigma = torch.exp(sigma)
        dist = Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z)
        return action.item()
    
    def get_action_vec(self, inputs):
        dist_param = self.actor(inputs)
        mus, sigmas = self.embedding_mu(dist_param), self.embedding_sigma(dist_param)
        sigmas = torch.clamp(sigmas, self.min_sigma, self.max_sigma)
        sigmas = torch.exp(sigmas)
        dist = Normal(mus, sigmas)
        action_z = dist.rsample()
        action_vec = torch.tanh(action_z)
        log_prob = dist.log_prob(action_z) - torch.log(1 - action_vec.pow(2) + torch.FloatTensor([1e-7]).to(self.device))
        return 2*action_vec, log_prob


    def train(self, target_model, replay_buffer, alpha = 0.001, rou = 0.01, gamma = 0.98, batch_size = 32):
        s, a, r, s_next, done = replay_buffer.sample_batch(batch_size)
        q_val1, q_val2 = [self.critic_ls[i](torch.cat([s, a], -1)) for i in range(2)]
        a_next_vec, a_next_log_prob = self.get_action_vec(s_next)
        q_target = r + gamma * (torch.min(torch.cat([target_model.critic_ls[i](torch.cat([s_next, a_next_vec.detach()], -1)) for i in range(2)], -1), -1, keepdim = True)[0] - alpha * a_next_log_prob) * done
        critic_loss = ((q_target.detach() - q_val1) ** 2 + (q_target.detach() - q_val2) ** 2)
        critic_loss = critic_loss.mean()  # 1,2 critic
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
         
        a_vec, a_log_prob = self.get_action_vec(s)
        actor_loss = -(torch.min(torch.cat([self.critic_ls[i](torch.cat([s, a_vec], -1)) for i in range(2)], -1), -1, keepdim = True)[0] - alpha * a_log_prob)
        actor_loss = actor_loss.mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
    
        # 参数跟随
        for target, cur in zip(target_model.parameters(), model.parameters()):
            target = rou * target + (1 - rou) * cur


if __name__ == "__main__":
    SEED = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_EPOCHS = 100000
    LEARNING_RATE = 1e-3
    MEM_SIZE = 40000
    MAX_STEP = 200
    RENDER = False


    model = SAC(args = (3, 2, LEARNING_RATE, DEVICE)).to(DEVICE)
    target_model = SAC(args = (3, 2, LEARNING_RATE, DEVICE)).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = ReplayBuffer(args = (MEM_SIZE, DEVICE))

    env = NormalizedActions(gym.make("Pendulum-v0")) 
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    train_flag = False
    for epi_i in range(MAX_EPOCHS):
        s = env.reset()
        score = 0.
        done = False
        while not done:
            if RENDER:
                env.render()
            action = model.choose_action(s)
            s_next, r, done, info = env.step(action)

            score += r
            replay_buffer.save_trans((s, action, r, s_next, done))
            if len(replay_buffer.buffer) > 200:
                train_flag = True
                model.train(target_model, replay_buffer)
            if done: 
                print("Epoch:{}   score:{}   training:{}".format(epi_i + 1, score, train_flag))
