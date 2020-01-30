# -#- coding: utf-8 -*-

"""
인간 플레이어 인터페이스와 관련된 클래스 모음
"""

__author__ = "Hyunsoo Park, Game AI Team, NCSOFT"

import pygame
import numpy as np
import chess


""" 일반 체스 좌표와 마이크로 체스 좌표를 매핑 """
std_to_micro = {
    32: 16, 33: 17, 34: 18, 35: 19,
    24: 12, 25: 13, 26: 14, 27: 15,
    16:  8, 17:  9, 18: 10, 19: 11,
    8:  4,  9:  5, 10:  6, 11:  7,
    0:  0,  1:  1,  2:  2,  3:  3,
}

""" 마이크로 체스 좌표와 일반 체스 좌표를 매핑 """
micro_to_std = {idx: square for square, idx in std_to_micro.items()}

""" 체스 기물 이미지 파일 경로 """
piece_dict = dict(K='scripts/data/Chess_klt60.png', Q='scripts/data/Chess_qlt60.png', R='scripts/data/Chess_rlt60.png',
                  B='scripts/data/Chess_blt60.png', N='scripts/data/Chess_nlt60.png', P='scripts/data/Chess_plt60.png',
                  k='scripts/data/Chess_kdt60.png', q='scripts/data/Chess_qdt60.png', r='scripts/data/Chess_rdt60.png',
                  b='scripts/data/Chess_bdt60.png', n='scripts/data/Chess_ndt60.png', p='scripts/data/Chess_pdt60.png')


class ChessBoard(object):
    """
    인간 플레이어 인터페이스 시각화 클래스
    
    :속성 block_size: int, 체스 한 칸의 픽셀 크기
    :속성 width: int, 마이크로 체스 가로 칸 수
    :속성 height: int, 마이크로 체스 세로 칸 수
    """
    block_size = 62
    width = 4 * block_size
    height = 5 * block_size

    def __init__(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Microchess')

        # background
        # Create the PixelArray.
        self.background = pygame.Surface((self.width, self.height))
        ar = pygame.PixelArray(self.background)
        r, g, b = 0, 0, 0

        for x in range(self.width):
            for y in range(self.height):
                pile, rank = x // self.block_size, y // self.block_size
                if (pile + rank) % 2 == 0:
                    r, g, b = 209, 139, 71
                else:
                    r, g, b = 255, 206, 158
                ar[x, y] = (r, g, b)
        del ar

        # pieces
        self.piece_imgs = dict()
        for p in 'KQRBNPkqrbnp':
            self.piece_imgs[p] = pygame.image.load(piece_dict[p])

        # initialize state
        self.board_state = np.zeros((5, 4), dtype=str)
        self.src_pos = None

    def step(self, board, move=None):
        """
        board 객체와 move 객체를 입력받아,
        pygame 모듈을 사용해서 상태를 시각화 함
        
        - AI 플레이어가 Move 를 입력하면, 단순히 현재 상태를 시각화 시킴고 그대로 Move를 반환
        - 인간플레이어는 Move 객체대신 None을 입력하고, 이 함수에서 마우스 입력을 받아, 
          게임 상태를 변경할 Move 객체를 생성해서 반환함
        - 폰을 마지막 칸으로 보내면 무조건 퀸으로 승급하도록 했음 (AI는 어떤 기물로든 승급가능)

        :param State board:
        :param chess.Move move:
        :return: (chess.Move) --
        """
        self._update_board_state(board)
        legal_moves = list(board.legal_moves)
        bksize = self.block_size

        screen = pygame.display.get_surface()
        screen.blit(self.background, (0, 0))
        for x in range(4):
            for y in range(5):
                if self.board_state[y, x] in 'KQRBNPkqrbnp':
                    pos = x + 4*y
                    self._draw_piece(screen, self.piece_imgs[self.board_state[y, x]], pos)

        while True:
            for m in legal_moves:
                pos1, pos2 = std_to_micro[m.from_square], std_to_micro[m.to_square]
                x1, y1 = pos1 % 4, 4 - (pos1 // 4)
                x2, y2 = pos2 % 4, 4 - (pos2 // 4)
                xy1 = [int(bksize * (x1 + 0.5)), int(bksize * (y1 + 0.5))]
                xy2 = [int(bksize * (x2 + 0.5)), int(bksize * (y2 + 0.5))]
                
                if self.src_pos == pos1:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
                pygame.draw.line(screen, color, xy1, xy2, 1)
                pygame.draw.circle(screen, color, xy2, 3, 1)

            pygame.display.flip()

            if move is None:
                # 인간 플레이어 입력
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos[0] // self.block_size, 4 - event.pos[1] // self.block_size
                    pos = x + 4*y
                    if self.src_pos is None:
                        if self.board_state[y, x] in 'KQRBNPkqrbnp':
                            moves = [m for m in legal_moves if std_to_micro[m.from_square] == pos]
                            if len(moves) > 0:
                                self.src_pos = pos
                    else:
                        dst_pos = pos
                        moves = [m for m in legal_moves
                                 if std_to_micro[m.from_square] == self.src_pos and
                                 std_to_micro[m.to_square] == dst_pos]

                        src_pos = self.src_pos
                        self.src_pos = None
                        if len(moves) > 0:
                            # 폰이 마지막 칸에 도달했는지 확인하고 Move 객체에 승급 세팅함
                            piece_is_pawn = False
                            reach_opposite_side = False

                            if board.turn is True:
                                if board.piece_at(micro_to_std[src_pos]).symbol() == 'P':
                                    piece_is_pawn = True
                                    if dst_pos in [16, 17, 18, 19]:
                                        reach_opposite_side = True
                            else:
                                if board.piece_at(micro_to_std[src_pos]).symbol() == 'p':
                                    piece_is_pawn = True
                                    if dst_pos in [0, 1, 2, 3]:
                                        reach_opposite_side = True

                            if reach_opposite_side and piece_is_pawn:
                                move = chess.Move(
                                    from_square=micro_to_std[src_pos], 
                                    to_square=micro_to_std[dst_pos],
                                    promotion=5)  # queen
                                return move
                            else:
                                move = chess.Move(
                                    from_square=micro_to_std[src_pos], 
                                    to_square=micro_to_std[dst_pos])
                                return move
            else:
                # AI 플레이어 입력,
                # AI 플레이어는 자신의 chess.Move 객체
                return move

    def _draw_piece(self, screen, piece, pos):
        pile, rank = pos % 4, pos // 4
        x = self.block_size * pile
        y = self.block_size * (4 - rank)
        screen.blit(piece, (x, y))

    def _update_board_state(self, board):
        self.board_state.fill('.')
        ranks = board.fen().split(' ')[0].split('/')
        x, y = 0, 0
        for rank in ranks[3:]:
            for p in rank:
                if p in 'KQRBNPkqrbnp':
                    self.board_state[y, x] = p
                    x += 1
                    if x > 4:
                        break
                else:
                    x += int(p)
            x = 0
            y += 1
        self.board_state[:] = np.flip(self.board_state, 0)
                    

if __name__ == '__main__':
    
    cb = ChessBoard()
    board = chess.Board()
    for i in range(100):
        if i % 2 == 0:
            move = cb.step(board)
        else:
            from random import choice
            move = cb.step(board, choice(list(board.legal_moves)))
        board.push(move)

