# -*- coding: utf-8 -*-
"""
Random AI 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import random
from agents import BaseAgent
    

class RandomAgent(BaseAgent):
    """
    Random AI

    - 무작위로 행동을 결정하는 예제
    """

    def reset(self):
        pass

    def act(self, state):
        moves = list(state.legal_moves)
        return random.choice(moves)

    def close(self):
        pass

