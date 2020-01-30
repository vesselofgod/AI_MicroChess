# -*- coding: utf-8 -*-
"""
scripts/run_game.py

Microchess AI 플랫폼에서 사용하는 체스와 관련된 기본 구성 요소 모음
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import argparse
import logging
import os
import sys
sys.path.append(os.path.abspath('.'))
import time
import importlib

import chess
import chess.svg
import numpy as np
from IPython.display import SVG, display
from scripts.chess_board import ChessBoard


logging.basicConfig(format='[%(asctime)-15s %(levelname)s] %(message)s')
logger = logging.getLogger('Game')
logger.setLevel(logging.DEBUG)

TIME_LIMIT = 400
"""
한 플레이어에게 주어진 시간제한,
한턴에 10초씩 80턴으로 계산해서 한 플레이어에 400초 할당
"""


class State(chess.Board):
    """
    마이크로 체스 AI 플랫폼에서 게임 상태를 전달하기 위해 사용

    - chess.Board + forward 기능
    - chess.Board 에서 제공하는 대부분의 기능은 그대로 사용가능
    - 향후 부정행위에 사용할 수 있는 기능이 발견되는 경우 일부 기능이 제한될 수 있음
    """
    white_remain_sec = TIME_LIMIT
    "(int) -- white 플레이어의 남은 시간"
    black_remain_sec = TIME_LIMIT
    "(int) -- black 플레이어의 남은 시간"

    def forward(self, move):
        """
        시뮬레이션
        - 현재 State에 move를 적용한 다음 상태를 반환
        
        :param Move move:
        :return: (:class:`State`) -- move가 적용된 다음 상태
        """
        next_state = self.copy(stack=True)
        next_state.push(move)
        return next_state

    @property
    def color(self):
        return self.turn


class Move(chess.Move):
    pass


class Environment(object):
    """
    마이크로 체스 게임 환경의 기본 객체

    - 게임의 초기화, 진행, 종료에 관한 기능은 담당
    - 문자를 이용한 기초적인 시각화 가능
    - 그 이상의 시각화 기능은 chess_board.py (pygame)와 visdom에서 담당
    """

    def __init__(self):
        self.human_play = 'console'
        self.viz = None
        self.board_win = None
        self.text_win = None
        self._last_action = None
        self.max_turns = None
        self.turn = 1
        self.board = None
        self.reset()

    def reset(self, fen=None, max_turns=80):
        """
        환경 초기화

        :param str fen: fen 표기법으로 체스 보드 상태를 초기화 함, None 으로 전달할 경우 기본 초기화 상태로 세팅함
        :param int max_turns: 최대 게임 턴, 기봅값 80
        :return: (State) -- 현재 게임 상태 반환
        """
        if fen:
            self.board = State(fen)
        else:
            self.board = State()
        self._last_action = None  # 시각화를 위해 필요함
        self.max_turns = max_turns
        self.turn = 1
        return self.board

    def render(self):
        """
        시각화

        - 언제나 jupyter console에 출력 가능한 svg객체를 출력함 (jupyter console에서 시각화 됨)
        - 문자열을 이용해 체스 상태를 출력함 (대문자: white, 소문자: black)
        - 인간 플레이어가 한명이라도 게임을 할 때는 pygame을 이용함
        """
        if self.human_play:
            self.cb = ChessBoard()

        display(SVG(chess.svg.board(self.board, lastmove=self._last_action)))
        print('\n'.join([line[:7] for line in str(self.board).split('\n')[3:]]))

    def step(self, move):
        """
        현재 상태에 move를 적용하여 다음 턴으로 게임을 진행
        
        :param move: chess.Move, 현재 플레이어의 다음 수
        :return: (:class:`State`, float, bool, dict) --

           - State, move가 적용된 다음 게임 상태
           - float, 보상, 이번 수로 게임이 승리하면 1., 패배하면 0., 그 외에는 0.5
           - bool, 이번 수로 게임이 종료되었는지 여부
           - dict, 그 외 기타 정보 전달 용
        """
        if self.human_play:
            move = self.cb.step(self.board, move)
        self.board.push(move)
        self._last_action = move
        reward = [0., 0.]
        done = False
        draw = False

        if self.board.is_game_over():
            done = True
            reward = self.board.result().split('-')
            for i, r in enumerate(reward):
                if r == '0':
                    reward[i] = 0.0
                elif r == '1':
                    reward[i] = 1.0
                elif r == '1/2':
                    reward[i] = 0.5
                    draw = True
                else:
                    raise Exception
        elif self.turn >= self.max_turns:
            done = True
            draw = True

        if draw:
            # 게임이 비기는 경우 평가값을 반올림하여 승=1, 패=0 점으로 변환
            # 기물의 점수까지 똑같은 경우에는 0.5
            reward = self._evaluate()
            for ri in range(2):
                reward[ri] = 0.5 if reward[ri] == 0.5 else round(reward[ri])

        self.turn += 1
        return self.board.copy(stack=True), reward, done, dict()

    def close(self):
        pass

    def _evaluate(self):
        """
        현재 상태(self.board)를 평가해서 white, black 순서로 평가값을 반환함,
        각 평가값은 [0, 1] 범위의 실수임, 두 수의 합은 1.0
        
        :return: ([float, float]) -- white 평가값, black 평가값
        """
        pieces = self.board.fen().split(' ')[0]
        black_score, white_score = 0, 0
        for p in pieces:
            if p == 'p':
                black_score += 1
            elif p == 'n' or p == 'b':
                black_score += 3
            elif p == 'r':
                black_score += 5
            elif p == 'q':
                black_score += 10
            elif p == 'k':
                black_score += 4
            elif p == 'P':
                white_score += 1
            elif p == 'N' or p == 'B':
                white_score += 3
            elif p == 'R':
                white_score += 5
            elif p == 'Q':
                white_score += 10
            elif p == 'K':
                white_score += 4

        norm_white_score = white_score / (white_score + black_score)
        norm_black_score = black_score / (white_score + black_score)
        return [norm_white_score, norm_black_score]

    
class Evaluator(object):
    """
    **기본 예제에서 사용하는 평가함수 모음**

    현재 상태(board.Chess)와 현재 턴(bool, white=True, black=False)를 입력으로 받아,
    평가 값을 [0, 1] 실수로 반환함
    """

    @staticmethod
    def win_or_lose(board, turn):
        """
        승/패만 판단하는 간단한 평가함수
        
        - 간단한 평가함수, [0., 0.5, 1.] 중 하나의 값을 반환
        - 승리하면 1., 패배하면 0.을 반환함
        - 나머지 경우에는 0.5 반환 (무승부, 게임이 진행 중)

        :param State board: 평가할 게임 상태
        :param bool turn: chess.WHITE 또는 chess.BLACK
        :return: (float) -- [0, 1] 범위 실수
        """
        if board.is_game_over():
            scores = board.result().split('-')
            for i, r in enumerate(scores):
                if r == '0':
                    scores[i] = 0.
                elif r == '1':
                    scores[i] = 1.
                elif r == '1/2':
                    scores[i] = 0.5
            if turn == chess.WHITE:
                return scores[0]
            elif turn == chess.BLACK:
                return scores[1]
        else:
            return 0.5

    @staticmethod
    def eval(board, turn):
        """
        기물 점수를 계산하는 평가함수

        - 현재 판위에 남아있는 기물의 종류와 개수에 따라 [0, 1] 실수값으로 평가함
        - 게임의 승패가 결정되어 있다면, Evaluator.win_or_lose로 평가함
        - 게임이 종료되지 않았다면 기물의 종류와 개수에 따라 점수를 계산한 뒤 [0, 1]값으로 정규화 시킴
        - 절대적인, 상대적인 기물의 위치, 캐슬링 여부등 다른 요소는 고려하지 않음

        기물의 점수

        - Pawn: 1점
        - Knight, Bissop: 3점
        - Rook: 5점
        - Queen: 10점
        - King: 4점

        :param State board: 평가할 상태
        :param bool turn: chess.WHITE 또는 chess.BLACK
        :return: (float) -- [0, 1] 범위 실수
        """
        if board.is_game_over():
            return Evaluator.win_or_lose(board, turn)
        else:
            pieces = board.fen().split(' ')[0]
            black_scores, white_scores = 0, 0
            for p in pieces:
                if p == 'p':
                    black_scores += 1
                elif p == 'n' or p == 'b':
                    black_scores += 3
                elif p == 'r':
                    black_scores += 5
                elif p == 'q':
                    black_scores += 10
                elif p == 'k':
                    black_scores += 4
                elif p == 'P':
                    white_scores += 1
                elif p == 'N' or p == 'B':
                    white_scores += 3
                elif p == 'R':
                    white_scores += 5
                elif p == 'Q':
                    white_scores += 10
                elif p == 'K':
                    white_scores += 4

            assert turn in [chess.WHITE, chess.BLACK]
            score = 0.
            if turn == chess.WHITE:
                score = white_scores / (white_scores + black_scores)
            elif turn == chess.BLACK:
                score = black_scores / (white_scores + black_scores)
            return score

    @staticmethod
    def eval_2(board, turn):
        """
        Evaluator.eval 결과([0, 1] 실수)를 [-1, 1] 실수로 변환

        :param State board: 평가할 상태
        :param bool turn: chess.WHITE 또는 chess.BLACK
        :return: (float) -- [-1, 1] 실수
        """
        return 2. * Evaluator.eval(board, turn) - 1.


def game(white, black, human_play, initial_fen, max_turns, timelimit):
    """
    게임을 1회 실행하는 함수

    :param str white: white 플레이어의 객체(BaseAgent 상속한 클래스) 경로
    :param str black: black 플레이어의 객체(BaseAgent 상속한 클래스) 경로
    :param bool human_play: 인간 플레이어용 인터페이스를 사용여부
    :param str initial_fen: 게임의 초기상태를 fen으로 전달
    :param int max_turns: 최대 게임 턴, 경진대회 기본은 80
    :param int timelimit: 최대 게임 시간, 경진대회 기본은 400
    :return: (:class:`State`, int, float) -- state, turn, reward
    """
    logger.debug('{} vs. {}'.format(white, black))
    agents = list()
    for turn, agent_path in [(chess.WHITE, white), (chess.BLACK, black)]:
        try:
            # 입력 받은 에이전트 클래스로 경로에서 클래스를 찾아서 에이전트를 생성함  
            module, name = agent_path.rsplit('.', 1)
            agent = getattr(importlib.import_module(module), name)(agent_path, turn)
            if turn is False:
                agent.level = 1
            agent.reset()
            agents.append(agent)
        except (ValueError, AttributeError) as exc:
            # 에이전트 생성이 실패함
            import traceback
            traceback.print_exc()
            color = 'White' if turn else 'Black'
            logger.error('Import {} agent module ({}) failure'.format(color, agent_path))
            logger.error('{}'.format(exc))
            exit(1)

    remaining_time = [timelimit, timelimit]
    env = Environment()
    env.human_play = human_play
    state = env.reset(initial_fen)

    for turn in range(max_turns):
        agent = agents[turn % 2]
        color = 'White' if agent.turn else 'Black'
        logger.debug('{}: {}'.format(color, agent))
        logger.debug('{}: board value {:.3f}'.format(
            color, Evaluator.eval(state, agent.turn)))
        env.render()
        try:
            # AI의 의사결정이 필요한 부분
            start_time = time.perf_counter()
            state.white_remain_sec = min(1, remaining_time[0])
            state.black_remain_sec = min(1, remaining_time[1])
            move = agent.act(state.copy())  # 반드시 상태를 복사해서 전달
            elapsed_time = time.perf_counter() - start_time
            remaining_time[turn % 2] -= elapsed_time
            logger.debug('{}: {:.1f} sec. remain'.format(color, remaining_time[turn % 2]))
            if remaining_time[turn % 2] < 0:
                # AI에게 주어진 사간이 초과한 경우 예외 발생
                logger.warning('{}: {} timeout'.format(color, str(agent)))
                raise TimeoutError
        except Exception as exc:
            # AI 에서 문제가 발생하면 에러메시지를 출력하고, 해당 AI의 패배로 루프 종료
            import traceback
            traceback.print_exc()
            logger.error('{}'.format(exc))
            logger.warning('{}: {} is mis-qualified'.format(color, str(agent)))
            winner = agents[(turn + 1) % 2]
            reward = np.zeros(2)
            reward[(turn + 1) % 2] = 1.
            reward[turn % 2] = 0.
            logger.debug('{} Win'.format('White' if winner.turn else 'Black'))
            break

        logger.debug('{}: move {}'.format(color, move))
        state, reward, done, info = env.step(move)
        logger.debug('{}'.format(state.fen()))
        if done:
            logger.debug(str(reward))
            if reward[0] == reward[1]:
                logger.debug('draw')
            else:
                winner = agents[0] if reward[0] > reward[1] else agents[1]
                logger.debug('{} Win'.format('White' if winner.turn else 'Black'))
            break

    env.render()
    env.close()
    [agent.close() for agent in agents]
    logger.debug('Score: [{:.3f} {:.3f}]'.format(reward[0], reward[1]))
    return state, turn, reward


if __name__ == '__main__':

    """
    선택 가능한 Agent 목록
    """
    agents = dict(
        human='agents.basic.human.Player',
        malfunction='agents.basic.debug_agent.MalfunctionAgent',
        first_move='agents.basic.debug_agent.FirstMoveAgent',
        random='agents.basic.random_agent.RandomAgent',
        one_step_search='agents.search.one_step_search_agent.OneStepSearchAgent',
        two_step_search='agents.search.two_step_search_agent.TwoStepSearchAgent',
        greedy='agents.search.two_step_search_agent.TwoStepSearchAgent',
        negamax='agents.search.negamax_search_agent.NegamaxSearchAgent',
        abp_negamax='agents.search.abp_negamax_search_agent.ABPNegamaxSearchAgent',
        mcts='agents.search.mcts_agent.MCTSAgent',
        mcts_dev='agents.search.mcts_agent.MCTSAgentDev',
        self_learning='agents.self_learning.agent.Agent',
        stockfish='agents.stockfish.agent.Stockfish',
    )

    parser = argparse.ArgumentParser(description='Chess AI platform')
    parser.add_argument('--white', type=str, default='random',
                        help='White agent 이름 또는 모듈 경로 '
                             'eg. {}'.format(', '.join(list(agents.keys()))))
    parser.add_argument('--black', type=str, default='random',
                        help='Black agent 이름 또는 모듈 경로 '
                             'eg. {}'.format(', '.join(list(agents.keys()))))
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='두 에이전트끼리 게임을 플레이하여 성능을 평가함'
                             'n-games (기본값 10) 만큼 게임을 플레이하고,'
                             ' 흑백을 바꿔서 다시 n-games만큼 플레이함')
    parser.add_argument('-N', '--n-games', type=int, default=10,
                        help='benchmark 시에 몇 게임을 플레이할 것인가')
    parser.add_argument('--timelimit', type=int, default=400,
                        help='Agent에게 주어지는 시간, 이 시간을 초과하면 실격패 함')
    parser.add_argument('--max-turns', type=int, default=80, 
                        help='최대 게임 턴, 이것을 초과하면 게임을 종료하고, '
                             '평가값에 따라 승패를 판단함')
    args = parser.parse_args()
    fen = None
    # fen을 특정 상태로 설정하면, 초기상태를 변경할 수 있음
    # fen = "8/8/8/1k6/R7/1B1N4/8/3K4 w - - 3 8"
    white_agent_path = agents.get(args.white, args.white)
    if not white_agent_path.startswith('agents.'):
        logger.error('Invalid agent path: {} ({})'.format(args.white, white_agent_path))
        exit(1)
    black_agent_path = agents.get(args.black, args.black)
    if not white_agent_path.startswith('agents.'):
        logger.error('Invalid agent path: {} ({})'.format(args. black, black_agent_path))
        exit(1)

    if args.benchmark is False:
        # 한 게임 플레이
        state, turns, result = game(white_agent_path, black_agent_path,
                                    True if 'human' in [args.white, args.black] else False,
                                    fen, max_turns=args.max_turns, timelimit=args.timelimit)

        logger.debug('turns: {}'.format(turns))
        logger.debug('Game end: {}'.format(state.is_game_over()))
        logger.debug('checkmate: {}'.format(state.is_checkmate()))
        logger.debug('Stalemate: {}'.format(state.is_stalemate()))
        logger.debug('Insufficient material: {}'.format(state.is_insufficient_material()))
        logger.debug('57 moves: {}'.format(state.is_seventyfive_moves()))
        logger.debug('5-fold: {}'.format(state.is_fivefold_repetition()))
        if result[0] > result[1]:
            logger.debug('White win: {}'.format(result))
        elif result[0] < result[1]:
            logger.debug('Black win: {}'.format(result))
        else:
            logger.debug('Draw: {}'.format(result))

    elif args.benchmark is True:
        # white, black agent로 args.n_games 게임 플레이하고,
        # white, black을 바꿔서 args.n_games 플레이
        # 총 2 * args.n_games 플레이하고 두 에이전트의 성능을 비교함
        # benchmark 실행결과는 benchmark-{white}-{black}.csv 파일에 기록함

        csv_filename = 'benchmark-{}-{}.csv'.format(args.white, args.black)
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'wt') as f:
                f.write('{},{},reverse\n'.format(args.white, args.black))

        wb_scores = np.zeros((args.n_games, 2))
        bw_scores = np.zeros((args.n_games, 2))

        for n in range(args.n_games):
            state, turns, result = game(white_agent_path, black_agent_path,
                                        True if 'human' in [args.white, args.black] else False,
                                        fen, max_turns=args.max_turns, timelimit=args.timelimit)
            wb_scores[n][:] = result[:]
            with open(csv_filename, 'at') as f:
                f.write('{:.3f},{:.3f},False\n'.format(result[0], result[1]))

            state, turns, result = game(black_agent_path, white_agent_path,
                                        True if 'human' in [args.white, args.black] else False,
                                        fen, max_turns=args.max_turns, timelimit=args.timelimit)
            bw_scores[n][0] = result[1]
            bw_scores[n][1] = result[0]
            with open(csv_filename, 'at') as f:
                f.write('{:.3f},{:.3f},True\n'.format(result[1], result[0]))

        logger.info('{:^24s} |{:^24s}'.format(args.white, args.black))
        for white_score, black_score in wb_scores:
            logger.info('W {: 22.3f} | B {: 22.3f}'.format(white_score, black_score))
        for black_score, white_score in bw_scores:
            logger.info('B {: 22.3f} | W {: 22.3f}'.format(black_score, white_score))

        wb_score_mean = wb_scores.mean(axis=0)
        logger.info('{} (White): {:.3f} vs. {} (Black): {:.3f}'.format(args.white, wb_score_mean[0], args.black, wb_score_mean[1]))
        bw_score_mean = bw_scores.mean(axis=0)
        logger.info('{} (Black): {:.3f} vs. {} (White): {:.3f}'.format(args.white, bw_score_mean[0], args.black, bw_score_mean[1]))
        score_mean = np.vstack([wb_scores, bw_scores]).mean(axis=0)
        logger.info('{}: {:.3f} vs. {}: {:.3f}'.format(args.white, score_mean[0], args.black, score_mean[1]))

