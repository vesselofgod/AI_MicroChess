# -*- coding: utf-8 -*-
"""
인간 플레이어 인터페이스를 작동시키기 위한 더미 AI
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

from agents import BaseAgent


class Player(BaseAgent):
    """
    Player

    인간 플레이어를 위한 더미 AI
    """
    
    def reset(self):
        pass

    def act(self, state):
        return None

    def close(self):
        pass

