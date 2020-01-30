# -*- coding: utf-8 -*-
"""
Stockfish Wrapper 
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import random
import platform
import chess
import chess.uci as chess_uci
from agents import BaseAgent


class Stockfish(BaseAgent):
    """
    UCI 인터페이스를 이용해 Stockfish를 래핑함, 실제 구현은 stockfish_8_*.exe 임

    Microchees에 맞게 Stockfish의 기능을 제한함

    - 탐색번위 제한: 5x4 번위 이외의 공간을 탐색하지 못하게 함
    - castling 제한: castling 위치가 다르기 때문에 Stockfish에서는 에러 발생
    
    TODO: 간헐적으로 에러가 발생함, 원인은 아직 불명
    """

    engine_path = "stockfish"
    engine = None
    # Microchess 에서 기물 배치 가능한 위치리스트
    available_squares = [
        32, 33, 34, 35,
        24, 25, 26, 27,
        16, 17, 18, 19,
         8,  9, 10, 11,
         0,  1,  2,  3]
    level = 20  # [1, 20] 사이의 정수, 높을 수록 강력함

    def reset(self):
        if 'windows' in platform.platform().lower():
            self.engine_path = './agents/stockfish/stockfish_8_x64.exe'

        self.engine = chess_uci.popen_engine(self.engine_path)
        self.engine.uci()
        self.engine.setoption({"Skill Level": self.level})

    def act(self, state):
        # fen 표기에서 castling 제거
        #   Microchess 플랫폼과 일반 chess 에서 castling 위치가 다르기 때문에
        #   stockfish 가 castling 하려고 하면 에러발생함
        fen = state.fen().split(' ')
        fen = ' '.join(fen[0:2] + ['-'] + fen[3:6])
        board = chess.Board(fen)

        self.engine.position(board)

        # Micorochess 플랫폼에서 가능한 수만 필터링
        available_moves = [m for m in state.legal_moves 
                           if m.to_square in self.available_squares]

        cmd = self.engine.go(searchmoves=available_moves)
        if cmd.bestmove is not None:
            return cmd.bestmove
        elif cmd.ponder is not None:
            return cmd.ponder
        else:
            return random.choice(available_moves)

    def close(self):
        self.engine.quit()
