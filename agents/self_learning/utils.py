# -*- coding: utf-8 -*-
"""
Self Learning AI에 필요한 기타 파일

"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import os
import logging
import chess
import torch
import time

import chess.svg
from sqlitedict import SqliteDict

from agents.self_learning import OBSERVATION_SHAPE
from agents.self_learning import LEN_STATE
from agents.self_learning import N_ACTIONS


def get_logger():
    """
    로거 반환

    :return: (logging.logger) --
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logger = logging.getLogger('SelfPlay')
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def make_piece_index():
    """
    알파벳으로 표시된 체스 기물을 정수로 변환하는 dict를 반환

    :return: (dict) -
    """
    code_book = dict()
    for code, char in enumerate('PRNBQKprnbqk.'):
        code_book[code] = char
        code_book[char] = code
    return code_book


piece_index = make_piece_index()


def encode_observation(state, turn, mirror):
    """
    fen 표기법으로 된 보드 상태를 입력받아서, numpy array 형태로 변환
    각 채널은 특정 말에 대응되며, 해당 말이 존재하는 위치에 1.로 표시
    나머지 공간은 0.

    :param str state: fen 표기법
    :param bool turn: white=True, black=False
    :param bool mirror: 미러 옵션 사용여부
    :returns: (torch.Tensor[6x8x8]) -- 게임 상태를 tensor로 변환
    """
    turn = int(turn)
    board = torch.zeros((12, 5, 4))
    str_board = str(chess.Board(state))
    for i, c in enumerate(str_board):
        if i % 2 == 0:
            code = piece_index[c]
            if code < 12:
                if mirror:
                    if turn == chess.WHITE:
                        c = code
                        y = int(i / 2 // 8) - 3
                        x = int(i / 2 % 8)
                    elif turn == chess.BLACK:
                        c = (code + 6) % 12
                        y = int((63 - i / 2) // 8)
                        x = int((63 - i / 2) % 8) - 4
                else:
                    c = code
                    y = int(i / 2 // 8) - 3
                    x = int(i / 2 % 8)
                board[c][y][x] = 1

    return board


# white 플레이어입장에서 일반 체스 좌표와 마이크로 체스 좌표를 변환하는 dict 생성
white_to_micro = {
    32: 16, 33: 17, 34: 18, 35: 19,
    24: 12, 25: 13, 26: 14, 27: 15,
    16:  8, 17:  9, 18: 10, 19: 11,
    8:  4,  9:  5, 10:  6, 11:  7,
    0:  0,  1:  1,  2:  2,  3:  3,
}
white_to_std = {idx: square for square, idx in white_to_micro.items()}


# black 플레이어입장에서 일반 체스 좌표와 마이크로 체스 좌표를 변환하는 dict 생성
black_to_micro = {
    3: 16,  2: 17,  1: 18,  0: 19,
    11: 12, 10: 13,  9: 14,  8: 15,
    19:  8, 18:  9, 17: 10, 16: 11,
    27:  4, 26:  5, 25:  6, 24:  7,
    35:  0, 34:  1, 33:  2, 32:  3,
}
black_to_std = {idx: square for square, idx in black_to_micro.items()}


def encode_move(move, turn, mirror):
    """
    chess.Move를 [0, 399] 정수로 변환, decode_move의 반대
    
    :param chess.Move move:
    :param bool turn: white=True, black=False
    :param bool mirror:
    :return: (int) -- 정수 [0, 399] 범위
    """
    turn = int(turn)
    from_sq, to_sq = move.from_square, move.to_square
    if mirror:
        if turn == chess.WHITE:
            from_sq = white_to_micro[from_sq]
            to_sq = white_to_micro[to_sq]
        elif turn == chess.BLACK:
            from_sq = black_to_micro[from_sq]
            to_sq = black_to_micro[to_sq]
    else:
        from_sq = white_to_micro[from_sq]
        to_sq = white_to_micro[to_sq]
    return 5 * 4 * from_sq + to_sq


def decode_move(board, move_code, turn, mirror):
    """
    [0, 399] 정수로 된 수를 chess.Move로 변환, encode_move의 반대

    :param State board:
    :param chess.Move move_code:
    :param bool turn:
    :param bool mirror:
    :return: (chess.Move) --
    """
    turn = int(turn)
    reach_opposite_side = False
    piece_is_pawn = False
    from_sq, to_sq = move_code // 20, move_code % 20

    if mirror:
        if to_sq in [16, 17, 18, 19]:
            reach_opposite_side = True

        if turn == chess.WHITE:
            from_sq = white_to_std[from_sq]
            to_sq = white_to_std[to_sq]
        elif turn == chess.BLACK:
            from_sq = black_to_std[from_sq]
            to_sq = black_to_std[to_sq]

        assert board.piece_at(from_sq) is not None
        if board.piece_at(from_sq).symbol() in 'Pp':
            piece_is_pawn = True
    else:
        if board.piece_at(from_sq).symbol() == 'P':
            piece_is_pawn = True
            if to_sq in [16, 17, 18, 19]:
                reach_opposite_side = True

        if board.piece_at(from_sq).symbol() == 'p':
            piece_is_pawn = True
            if to_sq in [0, 1, 2, 3]:
                reach_opposite_side = True

        from_sq = white_to_std[from_sq]
        to_sq = white_to_std[to_sq]

    if reach_opposite_side and piece_is_pawn:
        return chess.Move(from_sq, to_sq, promotion=5)  # queen
    else:
        return chess.Move(from_sq, to_sq)


def is_castling(state, turn, mirror=True):
    """
    castling 가능 여부 검사하여 실수 list로 반환, 게임 상태로 사용

    :param State state:
    :param bool turn:
    :param bool mirror:
    :return: ([float, float]) -- 길이 2
    """
    turn = int(turn)
    casting_info = state.fen().split(' ')[2]
    features = [0, 0]
    if mirror:
        if turn == 1:
            if 'K' in casting_info:
                features[0] = 1
            if 'k' in casting_info:
                features[1] = 1
        elif turn == 0:
            if 'k' in casting_info:
                features[0] = 1
            if 'K' in casting_info:
                features[1] = 1
    else:
        if 'K' in casting_info:
            features[0] = 1
        if 'k' in casting_info:
            features[1] = 1
    return features


class Record(object):
    """
    학습 동안에 저장되는 데이터를 모아두는 객체
    """
    def __init__(self, args):
        self.iteration = 0  # 현재 반복 횟수
        self.main_start_time = time.perf_counter()  # 학습 시작시간
        self.xs = list()
        self.wall_times = list()
        self.best_model_version = 0  # 현재 가장 좋은 모델 버전
        self.best_model_versions = list()
        self.eval_scores = list()
        self.opponent_level = list()
        self.eval_scores_to_opponent = list()
        self.train_policy_losses = list()
        self.train_value_losses = list()
        self.validation_policy_losses = list()
        self.validation_value_losses = list()
        self.total_losses = list()
        self.game_turns = list()
        self.learning_rate = args.learning_rate
        self.learning_rates = list()
        self.momentums = list()
        self.batch_sizes = list()
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.n_batches = args.n_batches


class Operators(object):
    """
    학습 도중 실험 조건을 변경하기 위해 사용
    """

    @staticmethod
    def lr(args, log, current_model, best_model, arguments=[]):
        try:
            c = float(arguments[0])
        except:
            c = 0.5
        old_value = args.learning_rate
        args.learning_rate = c * args.learning_rate
        new_value = args.learning_rate
        args.learning_rate = max(1e-10, args.learning_rate)
        logger.info('update lr: {:.10f} -> {:.10f}'.format(old_value, new_value))
        return args, log, current_model, best_model

    @staticmethod
    def n_simulations(args, log, current_model, best_model, arguments=[]):
        try:
            c = float(arguments[0])
        except:
            c = 1.0
        old_value = args.n_simulations
        args.n_simulations = int(c * args.n_simulations)
        new_value = args.n_simulations
        args.n_simulations = max(15, args.n_simulations)
        args.n_simulations = min(1500, args.n_simulations)
        logger.info('update n-simulations: {:.10f} -> {:.10f}'.format(old_value, new_value))
        return args, log, current_model, best_model


class Buffer(object):
    """
    학습에 필요한 메모리 공간을 미리 할당해두고 재사용하기 위해 사용
    """
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.observations = torch.zeros(self.batch_size, *OBSERVATION_SHAPE)
        self.states = torch.zeros(self.batch_size, LEN_STATE)
        self.moves = torch.zeros(self.batch_size, N_ACTIONS)
        self.legal_moves = torch.zeros(self.batch_size, N_ACTIONS)
        self.pi_prob = torch.zeros(self.batch_size, N_ACTIONS)
        self.values = torch.zeros(self.batch_size, 1)
        self.dones = torch.zeros(self.batch_size, 1)

        if args.cuda:
            self.observations = self.observations.cuda()
            self.states = self.states.cuda()
            self.moves = self.moves.cuda()
            self.legal_moves = self.legal_moves.cuda()
            self.pi_prob = self.pi_prob.cuda()
            self.values = self.values.cuda()
            self.dones = self.dones.cuda()

        self.policy_outs = torch.zeros(2, 1, self.batch_size, N_ACTIONS)