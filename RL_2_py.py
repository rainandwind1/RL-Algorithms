import torch
from torch import nn, optim
import torch.nn.functional as F
import gym
import numpy as np

class RL2(nn.Module):
    def __init__(self, input_size, output_size):
        super(RL2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 128
        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, 2, batch_first = False)
        self.fc_net = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def get_policy(self, inputs):
        s, a, r, done_flag, h_pre = inputs
        inputs = [s, a, r, done_flag]
        inputs = torch.tensor(input, dtype = torch.float32)
        a_p, h_next = self.gru(inputs, h_pre)
        policy = self.fc_net(a_p)
        action_prob =  F.softmax(policy)
        return action_prob, h_next



def train_net(model, lr, loss_fn)

if __name__ == "__main__":
    return 0

