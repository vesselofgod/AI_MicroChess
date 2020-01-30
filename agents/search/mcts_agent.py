# -*- coding: utf-8 -*-
"""
Monte-Carlo Tree Search 구현
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import sys
import time
import random
import logging
from math import sqrt
from math import log
from operator import attrgetter
from collections import namedtuple

from scripts.run_game import State
from scripts.run_game import Evaluator
from agents import BaseAgent


try:
    # visdom 설치된 경우 MCTS 탐색과정 시각화 가능함
    from visdom import Visdom
    import chess.svg
    from IPython.display import SVG, display

    visdom = Visdom()
    if not visdom.check_connection():
        # visdom 서버가 실행되고 있지 않으면 사용하지 않음
        visdom = None
                
except ImportError as exc:
    # visdom이 설치되어 있지 않음
    visdom = None


logging.basicConfig(format='[%(asctime)-15s] %(message)s')
logger = logging.getLogger('Game')


class MCTSAgent(BaseAgent):
    """
    MCTS AI

    평가용 MCTS AI,
    시뮬레이션 횟수를 최대한으로 하고, 탐색시간을 제한함,
    주어진 시간을 최대한 활용하여 최선의 성능을 보임,
    """
    planner = None
    n_simulations = 1000000
    search_time = 10
    max_depth = 6

    def reset(self):

        # MCTS Agent의 핵심기능은 MCTS planner에 있음
        self.planner = MCTSPlanner(self.turn,
                                   max_depth=self.max_depth,
                                   action_score='visits',
                                   eval_func=Evaluator.eval_2,
                                   reward_amplify=True,
                                   exploration_coef=1.0,
                                   debug=False)  # 평가용 버전은 시각화 기능 끔 -> 속도향상

        if self.planner.debug is True and visdom is not None:
            visdom.board_win = visdom.svg(SVG(chess.svg.board(chess.Board())).data,
                                          opts=dict(title='MCTS', width=850, height=850))

            visdom.tree_win = visdom.svg(chess.svg.tree(),
                                         opts=dict(title='Tree', width=850, height=850))

    def act(self, state):
        return self.planner.search(state.copy(), n_simulations=self.n_simulations, timeout=self.search_time)

    def close(self):
        pass


class MCTSAgentDev(BaseAgent):
    """
    개발용 MCTS AI

    시뮬레이션 횟수를 고정하고, 탐색 시간은 매우 크게함,
    PC 사양에관계없이 균일한 성능을 보임
    """

    planner = None
    # n_simulation 의 값이 클 수록 성능이 좋지만, 실행시간이 오래걸림
    # 마이크로 체스에서는 5000이면 stockfish 와 유사한 수즌의 성능을 보임
    n_simulations = 5000
    search_time = 10000
    # 최대 탐색 깊이
    max_depth = 6

    def reset(self):
        self.planner = MCTSPlanner(self.turn,
                                   max_depth=self.max_depth,
                                   action_score='visits',
                                   eval_func=Evaluator.eval_2,
                                   reward_amplify=True,
                                   exploration_coef=1.0,
                                   debug=True)  # 개발용 버전은 debug를 True로 설정함 -> 시각화 사용

        if self.planner.debug is True and visdom is not None:
            # 시각화 패널 초기 세팅
            visdom.board_win = visdom.svg(SVG(chess.svg.board(chess.Board())).data,
                                          opts=dict(title='MCTS', width=400, height=400))

            visdom.tree_win = visdom.svg(chess.svg.tree(),
                                         opts=dict(title='Tree', width=900, height=400))

    def act(self, state):
        return self.planner.search(state.copy(), n_simulations=self.n_simulations, timeout=self.search_time)

    def close(self):
        pass


class Node(object):
    """
    상태공간 노드

    Attrs:

    - (bool) -- color, white=True, black=False
    - (:class:`scripts.run_game.State`) -- state
    - (:class:`agents.search.macts_agent.Node`) -- parent 이전 상태를 저장하고 있는 노드
    - (dict) -- children, dict(key-> chess.Move, value-> Node), 특정 수를 두고 난 뒤의 상태
    - (int) -- visits, 이 노드를 방문한 횟수
    - (float) -- wins, 이 노드와 자식 노드에서 받은 보상의 총합
    - (float) -- ucb 이 노드의 ucb 값
    """

    # Node 객체에 아래 속성 이외의 속성을 추가하지 못함 (메모리 절약)
    __slots__ = ['turn', 'state', 'parent', 'children',
                 'ucb', 'wins', 'visits']
    
    def __init__(self, turn, state):
        self.turn = turn
        self.state = state
        self.parent = None
        self.children = dict()
        self.visits = 0
        self.wins = 0.
        self.ucb = -1

    @property
    def next_turn(self):
        return not self.turn
        
    def __repr__(self):
        return self._print_tree()
    
    def _print_tree(self, move=None, depth=0):
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


class Arrow(namedtuple('Arrow', ['tail', 'head', 'color', 'opacity', 'msg'])):
    """
    체스 보드 시각화에 사용
    """
    pass


class MCTSPlanner(object):
    """
    MCTS 알고리즘 구현
    """
    
    def __init__(self, color, max_depth=sys.maxsize, action_score='visits',
                 eval_func=Evaluator.eval_2, exploration_coef=1.0,
                 reward_amplify=True, debug=False):
        self.turn = color
        self.max_depth = max_depth
        self._depth = 0
        self.action_score = action_score
        self.eval_func = eval_func
        self.ucb_c = exploration_coef
        self.reward_amplify = reward_amplify
        self.memory = dict()
        self.debug = debug
        
    def search(self, state, n_simulations=1000, timeout=10):
        """
        주어진 상태와 제약조건(시뮬레이션 횟수와 시간제한)동안 시뮬레이션을 하고
        찾아낸 가장 좋은 수을 반환함
        
        :param State state: 현재 게임 상태
        :param int n_simulations: 최대 시뮬레이션 횟수
        :param int timeout: 시간제한 (sec.)
        :return: (chess.Move) -- 가장 좋은 수
        """
        v0 = Node(self.turn, state)
        # 이전에 탐색한 결과를 저장해 두고, 트리의 일부를 재사용
        v0 = self.memory.get(v0.key, v0)
        logger.debug('# of reuse nodes: {}'.format(v0.visits))
        self.memory.clear()

        start_time = time.perf_counter()
        simulation_count = 0
        while True:
            if simulation_count > n_simulations or time.perf_counter() - start_time > 0.99 * timeout:
                # 제약조건(시뮬레이션 횟수와 시간제한)을 초과하면 바로 시뮬레이션 종료
                color = 'White' if self.turn else 'Black'
                elapsed_time = time.perf_counter() - start_time
                simulation_per_sec = simulation_count / elapsed_time
                logger.debug('{}: Simulation speed: {:,.1f} (# of simulations: {:,}, elapsed {:.1f} sec.)'.format(
                    color, simulation_per_sec, simulation_count, elapsed_time))
                break

            self._depth = 0
            # 선택 및 확장
            vl = self.tree_policy(v0)
            # Monte-Carlo 시뮬레이션
            delta = self.default_policy(vl.state)

            # 보상신호 강화
            if self.reward_amplify:
                current_value = self.eval_func(v0.state, self.turn)
                if delta > current_value:
                    delta = 1.0
                elif delta < current_value:
                    delta = -1.0
                else:
                    delta = 0.0

            # 보상 역전파
            self.backup(vl, delta)
            simulation_count += 1

            if self.action_score == 'visits':
                max_visits = max([v.visits for v in v0.children.values()])

            if self.debug and visdom is not None:
                # 시각화 기능
                moves = list()
                title = 'MCTS //'

                if v0.children:
                    # max_value should be greater than 0.5
                    max_value = max([(v.wins/v.visits + 1) / 2 for v in v0.children.values()])
                    title += 'max_value={:.3f} //'.format(max_value)
                    max_visits = max([v.visits for v in v0.children.values()])
                    title += 'max_vists={:.3f} //'.format(max_visits)

                for move in v0.children:
                    value = v0.children[move].wins / v0.children[move].visits
                    value = (value + 1) / 2

                    color = 'k'
                    if v0.children[move].visits == max_visits:
                        color = 'b'
                    if value == max_value:
                        color = 'r'

                    msg = '{:.0f}'.format(value * 100)
                    if move in v0.children:
                        value = self.eval_func(v0.children[move].state, self.turn)
                        value = (value + 1) / 2
                        msg += ':{:.0f}'.format(value * 100)
                    moves.append(Arrow(move.from_square, move.to_square, color, value, msg))

                visdom.board_win = visdom.svg(
                    chess.svg.board(v0.state, arrows=moves),
                    win=visdom.board_win,
                    opts=dict(title=title, width=400, height=400))

                if simulation_count % int(n_simulations / 100) == 0:
                    visdom.tree_win = visdom.svg(
                                chess.svg.tree(v0, size=800),
                                win=visdom.tree_win,
                                opts=dict(title=title, width=900, height=400))
                
        return self.best_child(v0).action
    
    def tree_policy(self, node):
        """
        선택 및 확장 단계
        
        :param agents.search.macts_agent.Node node: 현재 노드 v0
        :return: (Node) -- 생성한 트리의 종단노드 vl
        """
        while not node.state.is_game_over() and self._depth < self.max_depth:
            untried = [action for action in node.state.legal_moves 
                       if action not in node.children.keys()]
            if untried:
                # 아직 시도해 보지 않은 수가 있다면 한번이라도 시도해봐야 함
                
                # 무작위 선택
                selected = untried[random.randint(0, len(untried)-1)]
                # 확장
                next_state = node.state.forward(selected)
                child = Node(node.next_turn, next_state)
                node.children[selected] = child
                child.parent = node
                node = child
                break
            else:
                # UCB 값 계산 및 선택
                for child in node.children.values():
                    child.ucb = child.wins / child.visits + self.ucb_c * sqrt(2 * log(node.visits) / child.visits)
                    child.ucb += random.gauss(0, 1e-6)
                node = max(list(node.children.values()), key=attrgetter('ucb'))

            self.memory[node.key] = node
            self._depth += 1

        return node
    
    def default_policy(self, state):
        """
        Monte Carlo 시뮬레이션
        
        :param State state: 탐색한 트리의 종단노드에 저장되어 있는 state
        :return: (float) -- 시뮬레이션 결과 얻은 최종 보상
        """
        while not state.is_game_over() and self._depth < self.max_depth:
            actions = list(state.legal_moves)
            action = actions[random.randint(0, len(actions)-1)]
            state = state.forward(action)
            self._depth += 1

        return self.eval_func(state, self.turn)
    
    def backup(self, node, delta):
        """
        시뮬레이션 결과 업데이트
        턴마다 Negamax 스타일로 보상의 부호를 바꿈, 
        보상구간 [-1, 1]을 가정함
        
        :param agents.search.macts_agent.Node node: 탐색한 트리의 종단노드
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
    
    def best_child(self, node):
        """
        현재 노드에서 가장 좋은 행동을 결정하여 반환함

        :param agents.search.macts_agent.Node node: 현재 상태 노드 v0
        :return: (chess.Move) -- 가장 좋은 행동
        """
        Result = namedtuple('Result', ['action', 'visits', 'wins', 'ucb', 'value'])
        moves = [Result(move, node.visits, node.wins, node.ucb, node.wins / node.visits)
                 for move, node in node.children.items()]
        best_move = max(moves, key=attrgetter(self.action_score))  # visits가 가장 높은 행동선택
        return best_move


if __name__ == '__main__':

    import chess
    from tqdm import trange
    my_color = chess.WHITE
    state = State()
    
    planner = MCTSPlanner(my_color)
    v0 = Node(my_color, state)

    for _ in trange(1000):
        planner._depth = 0
        vl = planner.tree_policy(v0)
        delta = planner.default_policy(vl.state)
        planner.backup(vl, delta)
    action = planner.best_child(v0)
    
    print(v0)
    print(action)
