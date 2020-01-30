# -*- coding: utf-8 -*-
"""
Two Step Search AI 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import logging
import random
from operator import itemgetter
from agents import BaseAgent
from scripts.run_game import Evaluator


logger = logging.getLogger('Game')
logger.setLevel(logging.DEBUG)


class TwoStepSearchAgent(BaseAgent):
    """
    Two Step Search Agent
    
    - 나의 다음 수와 상대방의 그 다음 수, 두 수를 시뮬레이션 하고 의사결정을 하는 AI
    - 게임 승패를 판단하는 것과 동일 한 평가함수 사용
    - 플랫폼의 다른 부분에서는 greedy AI로 부르기도 함
    - 예제에 포함된 두 번째로 약한 탐색기반 AI
    """

    def reset(self):
        pass

    def act(self, state):
        my_action_evals = list()
        for my_action in state.legal_moves:
            next_state = state.forward(my_action)
            if next_state.is_game_over():
                # 한 수 뒤에 게임이 종료될 경우 상대방의 수를 보지 않고 바로 평가
                # add_nodise: 평가값 이 동일한 경우 그 중에 하나를
                # 무작위로 선택하도록 하기 위해 작은 노이즈 추가
                score = Evaluator.eval(next_state, self.turn) + 1e-6 * random.random()
            else:
                score = self.opponent_act(next_state)
            my_action_evals.append((my_action, score))
            
        # 가장 평가값이 높은 수을 반환함
        best_action, score = max(my_action_evals, key=itemgetter(1))
        return best_action

    def opponent_act(self, state):
        """
        상대방의 행동 시뮬레이션
        이 상태의 평가는 상대방의 그 다음 행동에 따라 달라짐
        
        :param State state: 현재 상태
        :return: (float) -- 평가값 [0, 1] 범위
        """
        opp_action_evals = list()
        for opp_action in state.legal_moves:
            next_state = state.forward(opp_action)
            score = Evaluator.eval(next_state, self.turn) + 1e-6 * random.random()
            opp_action_evals.append((opp_action, score))
        _, score = min(opp_action_evals, key=itemgetter(1))
        return score

    def close(self):
        pass

