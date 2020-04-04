# Ornstein-Uhlenbeck
import numpy as np
import matplotlib.pyplot as plt
import torch

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == "__main__":
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))
    plt.figure()
    ou_ns = torch.tensor([[ou_noise() for i in range(4)]],dtype=torch.float32)
    print(ou_ns)
    y1 = []
    y2 = np.random.normal(0, 1, 1000)
    t = np.linspace(0, 100, 1000)
    for _ in t:
        y1.append(ou_noise())
    plt.plot(t, y1, c='r')
    plt.figure()
    plt.plot(t, y2, c='b')
    plt.show()
