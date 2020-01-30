
Chess AI Compeitition Platform
===============================

- 이 플랫폼은 Microchess를 플레이하는 AI 개발 경진대회에 사용하기 위한 목적으로 개발되었다.
- 이 플랫폼은 크게 Microchss 게임과 예제 AI들 그리고 실행 스크립트로 구성되어 있으며, Windows 10 환경에서 python 3.5 (anaconda)를 사용하여 개발 테스트 되어있다.
- 플랫폼이 python 으로 구현되어 있기 때문에 python 으로 AI를 구현하는 것이 권장되지만, 다른 언어로 구현한 AI를 사용할 수 도 있다.

빠른시작
--------

설치방법
~~~~~~~~~

이 문서에서는 플랫폼 개발환경과 같은 Windows 10 + python 3.5 (anaconda)를 기준으로 설명한다.
하지만, 대부분의 구성요소들이 순수 python 으로 구현되어 있기 때문에, 다른 운영체제에서 큰 문제없이 실행
가능하다.

**설치 순서**

1. 플랫폼 다운로드 및 압축해제
2. anaconda 다운로드 및 설치 (https://www.anaconda.com/download/)

3. 가상환경 생성 및 활성화::

   (base) C:\Users\user\MicrochessAICompetition> conda create –n mchess python=3.5
   (base) C:\Users\user\MicrochessAICompetition> activate mchess
   (mchess) C:\Users\user\MicrochessAICompetition>

4. 주요 모듈 설치::

   (mchess) C:\Users\user\MicrochessAICompetition> conda install numpy scipy ipython tqdm pyyaml mkl matplotlib
   (mchess) C:\Users\user\MicrochessAICompetition> conda install -c CogSci pygame
   (mchess) C:\Users\user\MicrochessAICompetition> pip install sqlitedict
   (mchess) C:\Users\user\MicrochessAICompetition> pip install visdom  # visdom 설치
   (mchess) C:\Users\user\MicrochessAICompetition> conda install -c peterjc123 pytorch-cpu=0.3.1  # pytorch 설치 (CPU, Windows 용)


AI 구현
~~~~~~~~~

AI를 구현할 때는 BaseAgent (agents/__init__.py)를 상속받아, reset, act, close 세 가지 메소드를 구현한다.
가장 간단한 예인 Random AI (agents/basic/random_agent.py)는 다음과 같이 구현한다.

.. literalinclude:: ../agents/basic/random_agent.py
   :language: python
   :lines: 7-26


이 세 가지 메소드는 플랫폼에의해 호출된다.

reset와 close는 AI의 초기화와 종료를 처리한다. 게임을 시작할 때와 끝낼 때 한번씩만 실행된다.
act는 AI의 턴마다 한번씩 실행된다. 현재 게임상태(state)를 입력받아, 다음 수(move)를 출력해야한다.

AI끼리 게임 플레이
~~~~~~~~~~~~~~~~~~~

AI끼리 게임을 플레이할 때는 scripts/run_game.py 을 사용한다. Random AI (white)와 One Step Search AI (black)끼리 플레이할 때는 다음과 같이 실행한다.::

   python scripts/run_game.py --white=agents.basic.random_agent.RandomAgent --black=agents.search.one_step_search_agent.OneStepSearchAgent

white와 black에 AI의 경로를 지정해 주면 게임을 바로 실행하고, 결과를 출력해 준다.

One Step AI는 다음 수의 결과를 고려하는 간단한 AI이다. 강력한 AI는 아니지만, Random AI는 쉽게 이길 수 있다.
실행한 결과는 다음과 같이 나타난다.

.. code-block:: none
   :linenos:

   [2018-04-23 15:40:07,079 DEBUG] White: agents.basic.random_agent.RandomAgent-True
   [2018-04-23 15:40:07,079 DEBUG] White: board value 0.500
   <IPython.core.display.SVG object>
   k n b r
   p . . .
   . . . .
   . . . P
   R B N K
   [2018-04-23 15:40:07,093 DEBUG] White: 400.0 sec. remain
   [2018-04-23 15:40:07,093 DEBUG] White: move b1d3
   [2018-04-23 15:40:07,093 DEBUG] 8/8/8/knbr4/p7/3B4/3P4/R1NK4 b Kk - 1 1
   ...  (생략)
   [2018-04-23 18:05:27,443 DEBUG] White: move a1a2
   [2018-04-23 18:05:27,443 DEBUG] 8/8/8/8/k7/8/K7/2b5 b - - 0 14
   [2018-04-23 18:05:27,443 DEBUG] [0, 1]
   [2018-04-23 18:05:27,443 DEBUG] Black Win
   <IPython.core.display.SVG object>
   . . . .
   k . . .
   . . . .
   K . . .
   . . b .
   [2018-04-23 18:05:27,449 DEBUG] Score: [0.000 1.000]
   [2018-04-23 18:05:27,449 DEBUG] turns: 26
   [2018-04-23 18:05:27,449 DEBUG] Game end: True
   [2018-04-23 18:05:27,449 DEBUG] checkmate: False
   [2018-04-23 18:05:27,449 DEBUG] Stalemate: False
   [2018-04-23 18:05:27,449 DEBUG] Insufficient material: True
   [2018-04-23 18:05:27,449 DEBUG] 57 moves: False
   [2018-04-23 18:05:27,449 DEBUG] 5-fold: False
   [2018-04-23 18:05:27,450 DEBUG] Black win: [0, 1]

4~8번째 줄 처럼 매 턴마다, 게임 상태를 텍스트 상태로 출력한다. white는 대문자로 표시하고, black은 소문자로 표시한다.

23번째 줄부터 게임 결과를 출력한 것이다. Score의 첫번째 숫자는 white의 점수를 나타내고, 두 번째 숫자는 black의 점수를 나타낸다.
(특별한 경우가 아니면 플랫폼의 나머지 부분에서도 white, black 순서로 출력함)
게임에 승리하면 1.0, 패배하면 0.0, 그리고 비기면 0.5점을 얻는다.

일반 Chess (Microchess 포함)와 달리 이 플랫폼에서는 승패가 결정되지 않으면,
게임이 종료되었을 때 남아있는 기물의 점수를 계산하여 최종 승패를 판단한다.
따라서, 기존 무승부 조건을 만족한 상태에서 기물의 점수도 동일한 경우만 무승부가 가능하다.

turns는 전체 게임의 턴 수를 보여주고, 그 다음 줄의 game end는 게임이 정상적으로 종료되었는지 여부를 알려준다.
그 다음부터는 게임이 종료된 이유(checkmate, stalemate, 등)를 알려준다.

AI 끼리의 성능을 평가할 때는 benchmark 옵션을 사용한다. benchmark 옵션을 사용하면, 지정된 white와 black 옵션과 상관없이
총 20게임 (지정된 white와 black으로 10게임, white와 black을 뒤집어서 10게임)을 플레이하고 전체 승률을 출력해준다.

AI와 게임 플레이
~~~~~~~~~~~~~~~~~

개발한 AI의 성능을 평가하기 위해 직접 게임을 플레이해봐야 할 필요가 있다.
다음과 같이 플랫폼을 실행하면 직접 게임을 플레이 해 볼 수 있다.::

   python scripts/run_game.py --white=human --black=one_step_search

.. _play_interface:
.. figure:: figs/microchess-human.png
   :figwidth: 200

   게임 플레이 인터페이스

일반 AI대신 human (agents.basic.human.Player)을 인자로 주면, 게임을 직접 플레이 할 수 있는 인터페이스 :ref:`play_interface` 가 활성화 된다.
