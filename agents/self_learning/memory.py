# -*- coding: utf-8 -*-
"""
Replay Memory 
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import os
import numpy as np
import torch
import random

from agents.self_learning.utils import get_logger
from agents.self_learning import OBSERVATION_SHAPE
from agents.self_learning import LEN_STATE
from agents.self_learning import N_ACTIONS


logger = get_logger()


class ReplayMemory(object):
    """
    게임 플레이 데이터를 저장하는 용도
    """
    def __init__(self, capacity, memory_path):
        """
        
        :param int capacity: 최대 저장가능한 데이터 샘플 개수
        :param str memory_path:
            - 메모리 파일을 저장할 경로, 기생성된 파일이 있으면 재사용한다.
            - 기생성된 파일과 데이터의 형태, capacity가 다른 경우 문제가 발생할 수 있으므로 주의
        """
        os.makedirs(memory_path, exist_ok=True)
        files = ['{}/observation.mem'.format(memory_path),
                 '{}/state.mem'.format(memory_path),
                 '{}/moves.mem'.format(memory_path),
                 '{}/legal_moves'.format(memory_path),
                 '{}/pis'.format(memory_path),
                 '{}/rewards'.format(memory_path),
                 '{}/dones'.format(memory_path)]
        if all([os.path.exists(p) for p in files]):
            # 이미 생성된 파일이 있는 경우 이전에 생성된 파일을 그대로 이용함
            mode = 'r+'
        else:
            mode = 'w+'
        
        logger.info('Open memory file: mode {}'.format(mode))

        self.meta = np.memmap('{}/meta.mem'.format(memory_path),
                              shape=3, dtype=np.int32, mode=mode)
        self.capacity = capacity if self.meta[0] == 0 else self.meta[0]
        self.capacity = min(capacity, self.capacity)
        self.size = self.meta[1]
        self.size = min(capacity, self.size)
        self.pos = self.meta[2]
        self.pos = self.pos % capacity

        self.observations = np.memmap('{}/observation.mem'.format(memory_path),
                                      shape=(capacity, *OBSERVATION_SHAPE), dtype=np.float32, mode=mode)
        self.states = np.memmap('{}/state.mem'.format(memory_path),
                                shape=(capacity, LEN_STATE), dtype=np.float32, mode=mode)
        self.moves = np.memmap('{}/moves.mem'.format(memory_path),
                               shape=(capacity, N_ACTIONS), dtype=np.float32, mode=mode)
        self.legal_moves = np.memmap('{}/legal_moves'.format(memory_path),
                                     shape=(capacity, N_ACTIONS), dtype=np.float32, mode=mode)
        self.pis = np.memmap('{}/pis'.format(memory_path),
                             shape=(capacity, N_ACTIONS), dtype=np.float32, mode=mode)
        self.rewards = np.memmap('{}/rewards'.format(memory_path),
                                 shape=capacity, dtype=np.float32, mode=mode)
        self.dones = np.memmap('{}/dones'.format(memory_path),
                               shape=capacity, dtype=np.int32, mode=mode)

    @property
    def capacity(self):
        return self.meta[0]

    @capacity.setter
    def capacity(self, value):
        self.meta[0] = value

    @property
    def size(self):
        return self.meta[1]

    @size.setter
    def size(self, value):
        self.meta[1] = value

    @property
    def pos(self):
        return self.meta[2]

    @pos.setter
    def pos(self, value):
        self.meta[2] = value

    def add_samples(self, samples):
        for sample in samples:
            self.observations[self.pos][:] = sample.observation
            self.states[self.pos][:] = sample.state
            self.moves[self.pos].fill(0)
            self.moves[self.pos][sample.move] = 1
            self.legal_moves[self.pos].fill(0)
            self.legal_moves[self.pos][sample.legal_moves] = 1
            self.pis[self.pos][:] = sample.pi
            self.rewards[self.pos] = sample.reward
            self.dones[self.pos] = sample.done
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def get_dataset(self, train_set_size, validation_set_size):
        """
        학습 데이터와, 검증 데이터의 개수를 입력함

        :param int train_set_size: 학습 데이터 개수
        :param int validation_set_size: 검증 데이터 개수
        :return: (torch.Tensor, torch.Tensor) -- train_set, validation_set
           - train_set: 학습 데이터 세트
           - validation_set: 검증 데이터 세트
        """
        assert self.size > (train_set_size + validation_set_size)
        idxs = random.sample(list(range(self.size)), train_set_size + validation_set_size)

        train_idxs = idxs[:train_set_size]
        observations = torch.from_numpy(self.observations[train_idxs])
        states = torch.from_numpy(self.states[train_idxs])
        moves = torch.from_numpy(self.moves[train_idxs])
        legal_moves = torch.from_numpy(self.legal_moves[train_idxs])
        pis = torch.from_numpy(self.pis[train_idxs])
        rewards = torch.from_numpy(self.rewards[train_idxs])
        dones = torch.from_numpy(self.dones[train_idxs])
        train_set = observations, states, moves, legal_moves, pis, rewards, dones

        validation_idxs = idxs[train_set_size:]
        observations = torch.from_numpy(self.observations[validation_idxs])
        states = torch.from_numpy(self.states[validation_idxs])
        moves = torch.from_numpy(self.moves[validation_idxs])
        legal_moves = torch.from_numpy(self.legal_moves[validation_idxs])
        pis = torch.from_numpy(self.pis[validation_idxs])
        rewards = torch.from_numpy(self.rewards[validation_idxs])
        dones = torch.from_numpy(self.dones[validation_idxs])
        validation_set = observations, states, moves, legal_moves, pis, rewards, dones
        return train_set, validation_set
