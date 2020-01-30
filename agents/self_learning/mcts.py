# -*- coding: utf-8 -*-
"""
Self Learning용 Monte Carlo Tree Search 구현



"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import time
import sys
import random
from operator import attrgetter
from math import sqrt
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import chess
import chess.svg
from torch.autograd import Variable

from agents.self_learning import N_ACTIONS
from agents.self_learning.utils import encode_observation
from agents.self_learning.utils import encode_move
from agents.self_learning.utils import decode_move
from agents.self_learning.utils import get_logger
from agents.self_learning.utils import is_castling


logger = get_logger()


class Node(object):
    """
    상태공간 노드

    Attrs:

    - (bool) -- turn, white=True, black=False
    - (:class:`scripts.run_game.State`) -- state
    - (:class:`agents.self_learning.mcts.Node`) -- parent, 이전 상태를 저장하고 있는 노드
    - (dict) -- children, dict(key-> chess.Move, value-> Node), 특정 수를 두고 난 뒤의 상태
    - (int) -- visits, 이 노드를 방문한 횟수
    - (float) -- wins, 이 노드와 자식 노드에서 받은 보상의 총합
    - (float) -- ucb, 이 노드의 ucb 값
    """
    __slots__ = ['turn', 'state', 'parent', 'children',
                 'ucb', 'wins', 'visits', 'prob']

    def __init__(self, turn, state):
        self.turn = turn
        self.state = state
        self.parent = None
        self.children = dict()
        self.visits = 0
        self.wins = 0.
        self.ucb = -1
        self.prob = None

    @property
    def next_turn(self):
        return not self.turn

    def __repr__(self):
        return self.print_tree()

    def print_tree(self, move=None, depth=0):
        """
        콘솔에 트리 출력
        """
        fmt = '{}:: {}-> v:{} vl:{:.3f} ({:.3f}) u:{:.3f} {}\n'
        buff = ''
        buff += '  ' * depth + fmt.format('White' if self.next_turn else 'Black',
                                          str(move),
                                          self.visits, self.wins,
                                          self.wins / (self.visits + 1e-6),
                                          self.ucb,
                                          self.state.fen())
        for move, c_node in self.children.items():
            buff += c_node.print_tree(move, depth + 1)
        return buff

    @property
    def key(self):
        return self.state.fen()


class MCTSPlanner(object):
    """
    MCTS 알고리즘 구현
    """

    def __init__(self, turn, max_depth=sys.maxsize, action_score='visits',
                 cputc=1., epsilon=0.2, alpha=0.8, model=None, mirror=True):
        self.turn = turn
        self.max_depth = max_depth
        self._depth = 0
        self.action_score = action_score
        self.cputc = cputc
        self.epsilon = epsilon
        self.alpha = alpha
        self.memory = dict()
        self.model = model
        self.verbose = False
        self.mirror = mirror

    def search(self, state, n_simulations=1000, tau=0., timeout=10):
        """
        주어진 상태와 제약조건(시뮬레이션 횟수와 시간제한)동안 시뮬레이션을 하고
        찾아낸 가장 좋은 수을 반환함

        :param State state: 현재 게임 상태
        :param int n_simulations: 최대 시뮬레이션 횟수
        :param int tau:
        :param int timeout: 시간제한 (sec.)
        :return: (chess.Move) -- 가장 좋은 수
        """
        v0 = Node(self.turn, state)
        # 이전에 탐색한 결과를 저장해 두고, 트리의 일부를 재사용
        v0 = self.memory.get(v0.key, v0)
        logger.debug('# of nodes reuse: {}'.format(v0.visits))
        self.memory.clear()
        prob, current_value = self.predict(state)
        v0.prob = prob

        start_time = time.perf_counter()
        simulation_count = 0
        while True:
            elapsed_time = time.perf_counter() - start_time
            if simulation_count > n_simulations or elapsed_time > 0.99 * timeout:
                # 제약조건을 초과하면 바로 시뮬레이션 종료
                color = 'White' if self.turn else 'Black'
                simulation_per_sec = simulation_count / elapsed_time
                logger.debug('{}: Simulation speed: {:,.1f} (# of simulations: {:,}, elapsed {:.1f} sec.)'.format(
                    color, simulation_per_sec, simulation_count, elapsed_time))
                break

            self._depth = 0
            # 선택 및 확장
            vl = self.tree_policy(v0)
            # Monte-Carlo 시뮬레이션(default policy) 생략
            #  --> tree_policy에서 인공신경망으로 대체
            delta = vl.wins
            # 보상 역전파
            self.backup(vl, delta)
            simulation_count += 1

        if tau == 0.:
            return self.best_action(v0, deterministic=True)
        else:
            return self.best_action(v0, deterministic=False)

    def tree_policy(self, node):
        """
        선택 및 확장 단계

        :param agents.self_learning.mcts.Node node: 현재 노드 v0
        :return: (Node) -- 생성한 트리의 종단노드 vl
        """
        while not node.state.is_game_over() and self._depth < self.max_depth:
            untried = [chess.Move(move.from_square, move.to_square) for move in node.state.legal_moves
                       if move not in node.children.keys()]
            if untried:
                # 아직 시도해 보지 않은 수가 있다면 한번이라도 시도해봐야 함

                # 무작위 선택
                selected = untried[random.randint(0, len(untried)-1)]
                # 확장
                next_state = node.state.forward(selected)
                child = Node(node.next_turn, next_state)
                # 인공신경망으로 행동 선택 확률과 현재 상태의 가치를 예측
                prob, value = self.predict(next_state)
                child.prob = prob
                child.wins = value
                child.visits += 1
                node.children[selected] = child
                child.parent = node
                node = child
                break
            else:
                # UCB 값 계산 및 선택
                for idx, (move, child) in enumerate(node.children.items()):
                    if self._depth == 0 and self.epsilon > 0.:
                        # [탐색] epsilon 값에 따라 무작위 행동 선택
                        eps = self.epsilon
                        nu = np.random.dirichlet([self.alpha] * len(node.children))
                    else:
                        # [활용] 최선의 행동을 선택
                        eps = 0.
                        nu = [0] * len(node.children)
                    Q = child.wins / child.visits
                    c = self.cputc
                    p = node.prob[encode_move(move, node.state.turn, self.mirror)]
                    U = c * ((1-eps)*p + eps*nu[idx]) * sqrt(node.visits) / (1 + child.visits)
                    n = random.gauss(0, 1e-6)
                    child.ucb = (Q + U + n).data[0]
                node = max(list(node.children.values()), key=attrgetter('ucb'))

            self.memory[node.key] = node

            self._depth += 1

        return node

    def predict(self, board_state):
        """
        인공신경망을 이용해 현재 상태에서 가능한 수들의 선택 확률과 가치를 예측함
        
        :param State board_state:
        :return: (torch.Tensor, torch.Tensor) -- move_prob, value
           - move_prob: torch.Tensor, 가능한 수들의 선택 확률
           - value: torch.Tensor, 현재 상태의 가치
        """
        observation = encode_observation(board_state.fen(), self.turn, self.mirror)
        observation = Variable(observation, volatile=True)
        state = torch.Tensor([int(board_state.turn), int(not board_state.turn)] + is_castling(board_state, board_state.turn, self.mirror))
        state = Variable(state, volatile=True)
        moves_logits, value = self.model(observation.unsqueeze(0), state.unsqueeze(0))

        move_mask = torch.zeros(N_ACTIONS)
        for move in board_state.legal_moves:
            move_mask[encode_move(move, board_state.turn, self.mirror)] = 1
        move_mask = Variable(move_mask, volatile=True)
        moves_logits = ((1 - move_mask) * -100) + (move_mask * moves_logits)  # non-legal moves ~ -100
        move_prob = F.softmax(moves_logits, dim=1)
        return move_prob.squeeze(), value.squeeze()

    def backup(self, node, delta):
        """
        시뮬레이션 결과 업데이트
        턴마다 Negamax 스타일로 보상의 부호를 바꿈,
        보상구간 [-1, 1]을 가정함

        :param agents.self_learning.mcts.Node node: 탐색한 트리의 종단노드
        :param float delta: 시뮬레이션으로 알아낸 보상
        """
        if node.parent.turn != self.turn:
            # Negamax: 상대방의 순서에서 종료되었을 때
            delta = -delta

        while node is not None:
            # 자식 상태부터 현재까지 보상값, 방문횟수 누적
            node.wins += delta
            node.visits += 1
            node = node.parent
            # Negamax
            delta = -delta

    def best_action(self, node, deterministic):
        """
        현재 노드에서 가장 좋은 행동을 결정하여 반환함

        :param agents.self_learning.mcts.Node node: 현재 상태 노드 v0
        :param bool deterministic: 결정적으로 행동을 선택(True)할 것인지 확률적으로 선택(False)할 것인지 여부
        :return: (chess.Move) -- 가장 좋은 행동
        """
        pi = self.pi(node, 1)

        if deterministic:
            # 가장 pi 값이 높은 행동 선택
            _, move_code = pi.max(0)
        else:
            # 확률적으로 선택
            move_code = pi.multinomial(1)
        return decode_move(node.state, move_code[0], node.state.turn, self.mirror), pi

    def pi(self, node, tau):
        """
        현재 상태 노드에서 pi 값을 계산함
        
        :param agents.self_learning.mcts.Node node:
        :param int tau:
        :return: (torch.Tensor[400]) -- 현재 노드에서 시뮬레이션 했던 모든 행동들을 확률 값으로 변환함
        """
        pi = torch.zeros(N_ACTIONS)
        for move, child in node.children.items():
            pi[encode_move(move, node.state.turn, self.mirror)] = pow(child.visits, 1/tau)
        return pi / pi.sum()
