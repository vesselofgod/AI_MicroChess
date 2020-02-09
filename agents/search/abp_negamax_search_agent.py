# -*- coding: utf-8 -*-
"""
Negamax Search  AI + alpha-beta pruning 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import time
import logging
import random
from agents import BaseAgent
from scripts.run_game import Evaluator


logger = logging.getLogger('Game')
logger.setLevel(logging.DEBUG)


class ABPNegamaxSearchAgent(BaseAgent):
    """
    Negamax Search AI + alpha-beta pruning

    - 기본 Negamax 알고리즘의 탐색 성능 향상을 위해 alpha-beta pruning 사용
    - alpha-beta pruning 은 불필요한 상태공간을 탐색하지 않기 때문에 같은 시간동안 더 깊은 탐색이 가능함
    """

    def reset(self):
        self.max_depth = 6  # 최대 탐색 깊이, alpha-bets pruning 덕분에 더 깊은 탐색이 가능함
        self.n_nodes = 0

    def act(self, state):
        start_time = time.perf_counter()
        self.n_nodes = 0
        color = 1. if self.turn else -1.
        # 초기 alpha 값은 -무한대, beta 값은 +무한대로 설정함
        alpha = -1000  # 내 순서에서 찾아낸 최대 보상
        beta = 1000  # 상대방의 순서에서 찾아낸 최소 보상
        best_move, best_value = self.negamax(state, self.max_depth, alpha, beta, color)
        logger.debug('# of searched nodes: {}'.format(self.n_nodes))
        logger.debug('etime: {:.1f} sec.'.format(time.perf_counter() - start_time))
        return best_move

    def negamax(self, state, depth, alpha, beta, color):
        if depth == 0 or state.is_game_over():
            heuristic_value = Evaluator.eval_2(state, True) + 1e-6 * random.random()
            return None, color * heuristic_value
        
        best_move, best_value = None, -1000
        for move in state.legal_moves:
            next_state = state.forward(move)
            _, value = self.negamax(next_state, depth-1, -beta, -alpha, -color)
            value = -value
            if value > best_value:
                best_move, best_value = move, value

            alpha = max(alpha, value)
            if alpha >= beta:
                # alpha가 beta보다 작지 않으면, 현재 깊이에서
                # 시뮬레이션하지 않은 나머지 행동은 할 필요가 없음
                break

        self.n_nodes += 1
        return best_move, best_value

    def close(self):
        pass
