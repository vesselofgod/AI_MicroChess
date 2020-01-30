# -*- coding: utf-8 -*-
"""
Self Learning AI 가 사용할 수 있는 인공신경망

"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class L1024r(torch.nn.Module):
    """
    1024개의 은닉노드를 가진 간단한 MLP
    """

    def __init__(self, observation_shape, len_state, n_actions):
        super(L1024r, self).__init__()
        self.observation_shape = observation_shape
        self.len_state = len_state
        self.n_actions = n_actions
        self.linear1 = nn.Linear(int(np.product(self.observation_shape)) + self.len_state, 1024)
        self.actor_linear = nn.Linear(1024, self.n_actions)
        self.critic_linear = nn.Linear(1024, 1)

        self.apply(weights_init)
        self.linear1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.train()

    def forward(self, observations, states):
        x = observations.view(-1, int(np.product(self.observation_shape)))
        x = torch.cat([x, states], dim=1)
        x = F.relu(self.linear1(x))
        return self.actor_linear(x), self.critic_linear(x)


class C64r6L1024r(torch.nn.Module):
    """
    CNN 6층 + MLP 1층 구조
    """
    
    def __init__(self, observation_shape, len_state, n_actions):
        super(C64r6L1024r, self).__init__()
        self.observation_shape = observation_shape
        self.len_state = len_state
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.linear1 = nn.Linear(64 * 5 * 4 + self.len_state, 1024)
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, self.n_actions)

        self.apply(weights_init)
        self.conv1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv2.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv3.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.linear1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.train()

    def forward(self, observations, states):
        x = self.conv1(observations)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = x.view(-1,  64 * 5 * 4)
        x = torch.cat([x, states], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        return self.actor_linear(x), self.critic_linear(x)
