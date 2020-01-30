# -*- coding: utf-8 -*-
"""
Negamax Search AI 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import time
import logging
import random
from agents import BaseAgent
from scripts.run_game import Evaluator


logger = logging.getLogger('Game')
logger.setLevel(logging.DEBUG)


class NegamaxSearchAgent(BaseAgent):
    """
    Negamax Search AI

    - Minimax AI의 일종인 Negamax AI 기본 구현
    - https://en.wikipedia.org/wiki/Negamax 참조
    - Negamax가 Minimax에 비해 코드가 간결함
    - Two Search AI의 일반적인 형태, Two Search AI와 달리 임의의 깊이를 탐색 가능함
    - 탐색 깊이가 깊어질 수록 탐색 공간이 10배씩 증가하기 때문에 최대 탐색 깊이를 적절히 선택해야함
    """

    def reset(self):
        self.max_depth = 4  # 최대 탐색 깊이, 가능한 한 턴에 10초 이하를 사용하는 범위
        self.n_nodes = 0  # 몇 개 노드를 탐색했는지 저장, 디버깅용

    def act(self, state):
        start_time = time.perf_counter()
        self.n_nodes = 0
        # color: white = 1, black = -1
        color = 1. if self.turn else -1.
        # 현재 상태에서 깊이 {max_depth}까지 탐색하고 가장 좋은 상태를 찾고, 그 항태와 연결된 수를 반환함
        best_move, best_value = self.negamax(state, self.max_depth, color)
        logger.debug('# of searched nodes: {}'.format(self.n_nodes))
        logger.debug('elapsed time: {:.1f} sec.'.format(time.perf_counter() - start_time))
        return best_move

    def negamax(self, state, depth, color):
        """
        Negamax 탐색
        
        :param State state: 현재 상태
        :param int depth: 남은 탐색 깊이, self.max_depth - 현재 깊이
        :param float color: white = 1., black = -1
        :return: (chess.Move, float) -- best_move, 평가값

        - chess.Move: 가장 좋은 행동
        - float: 가장 좋은 행동을 했을 때 얻을 수 있는 미래 보상
        """
        if depth == 0 or state.is_game_over():
            # 입력 받은 상태가 최대 탐색 깊이거나, 게임이 종료된 상태라면, 상태를 평가함
            heuristic_value = Evaluator.eval_2(state, True) + 1e-6 * random.random()
            # white는 1.0, black은 -1.0이 가장 좋은 점수
            return None, color * heuristic_value
        
        # 최저값으로 기본값 결정
        best_move, best_value = None, -1000
        for move in state.legal_moves:
            next_state = state.forward(move)
            # 재귀적으로 미래상태를 시뮬레이션함
            # 다음 상태는 상대방 순서이므로, color를 바꾸고, 
            # 반환받은 평가값(value)도 부호를 변경함
            _, value = self.negamax(next_state, depth-1, -color)
            value = -value
            if value > best_value:
                best_move, best_value = move, value
                
        self.n_nodes += 1
        return best_move, best_value

    def close(self):
        pass

