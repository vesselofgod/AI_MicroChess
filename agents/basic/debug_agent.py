# -*- coding: utf-8 -*-
"""
플랫폼 디버깅 및 테스트 용으로 만들어진 AI 모음
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import random
from agents import BaseAgent


class MalfunctionAgent(BaseAgent):
    """
    Malfunction Agent

    한 수를 둘 때마다 50%의 확률로 예외를 발생함
    
    - 플랫폼 테스트용
    - 예외를 발생시키면 반드시 패배해야함
    """

    def reset(self):
        pass

    def act(self, state):
        if random.random() < 0.5:
            raise Exception('Something is worng???')
        else:
            actions = list(state.legal_moves)
            return random.choice(actions)

    def close(self):
        pass
    

class FirstMoveAgent(BaseAgent):
    """
    First Move Agent

    가능한 수 중에서 가장 첫번째 수를 선택함

    - 플랫폼 테스트용
    """
    def reset(self):
        pass

    def act(self, state):
        actions = list(state.legal_moves)
        return actions[0]

    def close(self):
        pass
