# -*- coding: utf-8 -*-
"""

"""

__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


class BaseAgent(object):
    """
    AI의 기반 클래스
    AI를 구현할 때는 이 클래스를 상속받아 reset, act, close를 구현해야 함
    """

    def __init__(self, name, color):
        """
        :param str name: 출력용 AI 이름, 주요 logic 에서 사용하지 않음
        :param bool color:
           - True: chess.WHITE
           - False: chess.BLACK
        """
        self.name = name
        self.turn = color

    def reset(self):
        """
        AI 초기화
        게임 초기에 한 번 실행
        """
        pass

    def act(self, state):
        """
        AI 초기화
        
        :param State state: 현재 게임 상태
        :return: 다음 수 반환

            - :class:`chess.Move` -- AI의 다음수
            - None -- 다음 수를 반환할 필요가 없는 경우
        """
        raise NotImplementedError('"act" should be implemented')
        return None

    def close(self):
        """
        AI 종료
        마지막에 한 번 실행
        """
        pass

    @property
    def opponent_color(self):
        """
        상대의 턴
        """
        return not self.turn

    def __repr__(self):
        return '{}-{}'.format(self.name, 'White' if self.turn else 'Black')
