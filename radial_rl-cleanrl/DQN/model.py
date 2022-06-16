from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class MLP_QNetwork(nn.Module):
    def __init__(self, env):
        super(MLP_QNetwork, self).__init__()
        self.env = env
        self.num_actions = env.action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n)
        )
        self.train()

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        x = torch.reshape(x, (-1, np.array(self.env.observation_space.shape).prod()))
        return self.network(x)
    
    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0].item()
            else:
                action = random.randrange(self.num_actions)
        return action

class CnnDQN(nn.Module):
    def __init__(self, env, frames=4):
        super(CnnDQN, self).__init__()
        
        self.num_actions = env.action_space.n
        self.network = nn.Sequential(
            # Scale(1 / 255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            Linear0(512, env.action_space.n),
        )
        self.train()
        
    def forward(self, x):
        x = self.network(x.permute((0, 3, 1, 2)))
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0].item()
            else:
                action = random.randrange(self.num_actions)
        return action