# -*- coding: utf-8 -*-
"""
One Step Search AI 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import logging
import random
from operator import itemgetter
from agents import BaseAgent
from scripts.run_game import Evaluator


logger = logging.getLogger('Game')
logger.setLevel(logging.DEBUG)


class OneStepSearchAgent(BaseAgent):
    """
    One Step Search Agent

    - 현재 가능한 모든 수를 시뮬레이션 하여 다음 결과를 예상하고, 가장 좋은 수를 결정하는 AI
    - 바로 다음 단계만을 고려하는 가장 단순한 탐색 AI
    - 예제에 포함된 가장 약한 탐색 AI
    """

    def reset(self):
        pass

    def act(self, state):
        evals = list()
        for move in state.legal_moves:
            # simulation: 현재 상태(state)에서 특정 수(move)를 뒀을 때, 다음 상태(next_state) 예측
            next_state = state.forward(move)
            # 다음 상태(next_state)가 현재 플레이어(self.turn)에게 얼마나 좋은지 평가함
            value = Evaluator.eval(next_state, self.turn) + 1e-6 * random.random()
            evals.append((value, move))

        # 평가 값이 가장 높은 수 반환
        return max(evals, key=itemgetter(0))[1]

    def close(self):
        pass
