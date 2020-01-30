# -*- coding: utf-8 -*-
"""
Self Learning AI
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'

import os
import random
import sys
import time
import gc
import queue
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import chess
from scipy import stats
from tqdm import trange
from torch.autograd import Variable

from agents import BaseAgent
from agents.self_learning.memory import ReplayMemory
from scripts.run_game import Environment
from scripts.utils import ascii_plot
from agents.search.two_step_search_agent import TwoStepSearchAgent
from agents.stockfish.agent import Stockfish
from agents.self_learning.models import L1024r
from agents.self_learning.models import C64r6L1024r
from agents.self_learning.arguments import get_args
from agents.self_learning.utils import get_logger
from agents.self_learning.utils import encode_observation
from agents.self_learning.utils import encode_move
from agents.self_learning.utils import is_castling
from agents.self_learning.utils import Buffer
from agents.self_learning.utils import Record
from agents.self_learning.utils import Operators
from agents.self_learning.mcts import MCTSPlanner
from agents.self_learning import OBSERVATION_SHAPE
from agents.self_learning import LEN_STATE
from agents.self_learning import N_ACTIONS

# 시간절약을 위해 여러 게임을 동시에 실행시킬 것이기 때문에
# 선형대수 라이브러리들이 싱글 쓰레드로만 작동하도록 설정함
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP single thread 실행
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL single thread 실행

# 학습률 튜닝용
CMD_BUFFER = ''
CMD_QUEUE = list()

logger = get_logger()


class Agent(BaseAgent):
    """
    Self Learning AI

    초기 속성은 eval 모드에서 사용할 값으로 설정되어 있음
    """
    model = None

    model_cls = C64r6L1024r
    model_file = 'models/C64r6L1024r/v1/best.pt'
    max_depth = 32
    n_simulations = 100000

    cputc = 1.0
    epsilon = -1.0
    alpha = 1.0
    mirror = True
    planner = None

    def set_params(self, model, max_depth, n_simulations, cputc, epsilon, alpha, mirror):
        """
        AI의 초기화 작업은 reset 에서 실행하지만, 편의 목적으로 일부 값은 set_params 에서 설정함
        set_params 는 reset 이전에 호출되어야 함

        :param torch.nn model: 인공신경망(policy + value net)
        :param int max_depth: MCTS 최대 탐색 깊이
        :param int n_simulations: MCTS 시뮬레이션 횟수
        :param float cputc: MCTS, UCB 공식에서 U의 가중치 c (cU+Q)S
        :param float epsilon: MCTS, UCB 공식에서 탐색 가중치
        :param float alpha: MCTS, UCB 공식에서 활용 가중치
        :param bool mirror:
            True일 때는 학습 AI가 흑과 백의 데이터를 모두 학습에 사용함
            현재 자신과 다른 색깔일때는 판을 180도 회전하여 학습 데이터로 사용
        """
        self.model = model
        self.max_depth = max_depth
        self.n_simulations = n_simulations
        self.cputc = cputc
        self.epsilon = epsilon
        self.alpha = alpha
        self.mirror = mirror

    def reset(self, load_model=True, visualize_tree=False):
        """
        :param bool load_model:
           학습한 모델을 로드할 지 여부, 학습할 때는 False 로 하고,
           평가할 때는 기본 값 True 사용함
        :param bool visualize_tree:
           MCTS 시각화 기능을 사용하는지 여부, 학습할 때는 False, 평가할 때는 True
        """
        if load_model:
            self.model = self.model_cls(OBSERVATION_SHAPE, LEN_STATE, N_ACTIONS)
            self.model.load_state_dict(torch.load(self.model_file))

        self.planner = MCTSPlanner(self.turn,
                                   max_depth=self.max_depth,
                                   cputc=self.cputc,
                                   epsilon=self.epsilon,
                                   alpha=self.alpha,
                                   model=self.model,
                                   mirror=self.mirror)

    def act(self, state, include_pi=False, tau=0):
        """
        :param scripts.run_game.State state: 현재 게임 상태
        :param bool include_pi:
            다음 수와 pi를 같이 반환할지 여부
            True일 때는 pi를 같이 반환함
        :param int tau:
            몇 번째 수 까지 탐색을 할 것인지 [0, n]
            0 일 때는 탐색을 하지 않음 (평가, 테스트)
            0 이상일 때는 정해진 확률에 따라 무작위로 다음 수를 선택함 (학습)
        :return: (int, np.array[1, 400]) -- move, pi

           - move: [0, 399] 사이의 정수, 탐색을 하지 않는다면, 가장 pi값이 높은 행동을 선택함
           - pi: 1 x 400 실수 행렬, 모든 행동에 대한 MCTS의 탐색 빈도를 정규화
        """
        assert tau >= 0
        move, pi = self.planner.search(state.copy(), n_simulations=self.n_simulations, tau=float(tau))
        return (move, pi) if include_pi else move

    def close(self):
        pass


class Sample(object):
    """
    데이터 샘플 저장 용
    """
    __slots__ = ['observation', 'state', 'move', 'pi', 'reward', 'done', 'legal_moves']

    def __init__(self, observation, state, move, pi, reward, done, legal_moves):
        self.observation = observation
        self.state = state
        self.move = move
        self.pi = pi
        self.reward = reward
        self.done = done
        self.legal_moves = legal_moves

    def __repr__(self):
        return '{}, {}, {}, {}, {}'.format(
            self.observation, self.state, self.move, self.reward, self.done)


def play_game(inqueue, outqueue, seed):
    """
    종료되지 않고 무한 루프로 실행,
    inqueue 에서 다음 인자를 받아서, outqueue로 반환

    :param inqueue:
        - (Agent) -- white, white AI
        - (Agent) -- black, black AI
        - (float) -- turns_until_tau0, 탐색 횟수 (tau)
        - (int) -- max_turn, 최대 턴 (default: 80)
        - (bool) -- mirror
        - (bool) -- reversed_reward, 학습 AI가 black 일 때 True 설정
    :param outqueue:
        - (float) -- reward
        - (bool) -- turn
        - (list) -- short_term_memory
    :param int seed: worker 를 이 seed 값으로 초기화, worker 는 모두 다른 seed 값으로 초기화 해야 함
    """
    torch.manual_seed(seed)

    while True:
        try:
            white, black, turns_until_tau0, max_turn, mirror, reverse_reward = inqueue.get_nowait()

            agents = [white, black]
            short_term_memory = list()
            for agent in agents:
                if agent.__class__ == Agent:
                    agent.reset(load_model=False, visualize_tree=None)
                else:
                    agent.reset()

            env = Environment()
            env.human_play = None
            state = env.reset()
            remaining_time = [400, 400]

            for turn in range(max_turn):
                agent = agents[turn % 2]
                start_time = time.perf_counter()
                state.white_remain_sec = min(1, remaining_time[0])
                state.black_remain_sec = min(1, remaining_time[1])

                if agent.__class__ == Agent:
                    # 학습 Agent 턴
                    if turn < turns_until_tau0:
                        move, pi = agent.act(state.copy(), include_pi=True, tau=1)
                    else:
                        move, pi = agent.act(state.copy(), include_pi=True, tau=0)
                else:
                    # two step search 또는 stockfish 턴
                    try:
                        move = agent.act(state.copy())
                        pi = None
                    except Exception as exc:
                        import traceback; traceback.print_exc()
                        # 매우 간헐적으로 stockfish가 문제가 생기는 경우가 있음
                        # 이때는 전체 학습이 중단되는것을 막기 위해 stockfish의 수를 무작위로 결정함 
                        move = random.choice(list(state.legal_moves))
                        pi = None

                elapsed_time = time.perf_counter() - start_time
                remaining_time[turn % 2] -= elapsed_time
                legal_moves = [encode_move(lm, agent.turn, mirror) for lm in state.legal_moves]
                next_state, reward, done, _ = env.step(move)
                # 실수로 된 보상을 [0, 0.5, 1] 로 반올림 
                reward = [
                    reward[0] if reward[0] == 0.5 else round(reward[0]),
                    reward[1] if reward[1] == 0.5 else round(reward[1]),
                ]

                sample = Sample(encode_observation(state.fen(), agent.turn, mirror),
                                [int(agent.turn), int(not agent.turn)] + is_castling(state, agent.turn, mirror),
                                encode_move(move, agent.turn, mirror), pi, np.NaN, done, legal_moves)
                short_term_memory.append(sample)

                if done:
                    break

                state = next_state

            for i, sample in enumerate(short_term_memory):
                _reward = reward[0] if sample.state[0] == 1 else reward[1]
                _reward = 2. * _reward - 1.
                short_term_memory[i].reward = _reward

            if reverse_reward:
                reward = [reward[1], reward[0]]

            env.close()
            [agent.close() for agent in agents]
            outqueue.put((reward, turn, short_term_memory))

        except queue.Empty:
            time.sleep(1)


def train():
    """
    모델 학습 함수
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # model 초기화
    arch = globals().get(args.arch, None)
    if arch is None:
        logger.error("can't find proper NN class ({})".format(args.arch))
        exit(1)

    # n_best 개의 best 모델과 current 모델 생성, 가중치는 무작위로 생성
    best_models = [arch(OBSERVATION_SHAPE, LEN_STATE, N_ACTIONS) for _ in range(args.n_bests)]
    current_model = arch(OBSERVATION_SHAPE, LEN_STATE, N_ACTIONS)
    # 마지막 best 모델을 model 디스크에 저장하고, current 모델의 가중치를 이것으로 갱신
    torch.save(best_models[-1].state_dict(), '{}/best.pt'.format(args.out_dir))
    current_model.load_state_dict(best_models[-1].state_dict())

    # optimizer 초기화
    optimizer = optim.SGD(current_model.parameters(), args.learning_rate,
                          momentum=args.momentum, weight_decay=args.l2_penalty)
    # 반복횟수 100, 200, 300, 400, 500 때마다, 학습률을 0.5씩 감쇠함
    milestones = [100, 200, 300, 400, 500]
    lr_degration = 0.5

    # visdom 초기화, 반드시 필요
    from visdom import Visdom
    viz = Visdom(env=args.out_dir, port=args.port)
    viz.cmd_win = viz.text('<br>', opts=dict(title=args.date+':cmd', width=910, height=100))
    startup_sec = 1
    while not viz.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    assert viz.check_connection(), 'No connection could be formed quickly'


    def type_callback(event):
        """
        visdom widget, 텍스트 입력창
        """
        global CMD_BUFFER, CMD_QUEUE

        if event['event_type'] == 'KeyPress':
            if event['key'] == 'Enter':
                logger.info('command pending: {}'.format(CMD_BUFFER))
                CMD_QUEUE.append(CMD_BUFFER)
                CMD_BUFFER = ''
            elif event['key'] == 'Backspace':
                CMD_BUFFER = CMD_BUFFER[:-1]
            elif event['key'] == 'Delete':
                CMD_QUEUE = list()
                CMD_BUFFER = ''
            elif len(event['key']) == 1:
                CMD_BUFFER += event['key']
            content = '<br>'.join(CMD_QUEUE) + '<br>' + CMD_BUFFER
            viz.text(content, win=viz.cmd_win)

    try:
        # visdom 1.7.0 버전은 지원 안됨
        viz.register_event_handler(type_callback, viz.cmd_win)
    except:
        pass

    # 그 외 초기화
    record = Record(args)
    memory = ReplayMemory(args.max_memory_size, args.memory_path)
    logger.info('Memory path: {}'.format(args.memory_path))
    buff = Buffer(args)

    # worker process 초기화
    torch.manual_seed(args.seed)
    inqueue = mp.Queue()
    outqueue = mp.Queue()
    workers = [mp.Process(target=play_game, args=(inqueue, outqueue, args.seed + rank), daemon=True)
               for rank in range(args.n_workers)]
    [w.start() for w in workers]

    # 테스트 세팅
    # Level 0: Two Step Search AI, Level 1~20: Stockfish
    opponent_idx = 0
    opponents = [(TwoStepSearchAgent, 'two_step_search', None)]
    for level in range(1, 21):
        opponents.append((Stockfish, 'stockfish_lvl_{}'.format(level), level))

    try:
        for _ in range(args.n_iter):
            logger.info(str(args))
            # 현재 실험 조건창 갱신
            viz.args_win = viz.text('<p>{}</p><p>{}</p>'.format(' '.join(sys.argv), str(args)),
                                    win=viz.args_win if hasattr(viz, 'args_win') else None,
                                    opts=dict(title=args.date+': args', width=910, height=200))
            # 가능하다면, 명령창 갱신
            viz.cmd_win = viz.text('<br>'.join(CMD_QUEUE) + '<br>' + CMD_BUFFER,
                                   win=viz.cmd_win if hasattr(viz, 'cmd_win') else None,
                                   opts=dict(title=args.date+': cmd', width=910, height=100))

            # Self Learning 시작 
            logger.info('Start {} iteration'.format(record.iteration))
            loop_start_time = time.perf_counter()
            logger.info('Start selfplay')

            # best 모델을 평가모드로 바꾸고, cpu의 공유메모리에 복사
            for best_model in best_models:
                best_model.eval()
                best_model.cpu().share_memory()

            stime = time.perf_counter()
            # best 모멜을 무작위로 선택하여 서로 n_selfplay 만큼 게임을 함
            for _id in range(args.n_selfplay):
                white = Agent('White', chess.WHITE)
                white.set_params(random.choice(best_models), args.max_depth, args.n_simulations, args.cputc,
                                 args.epsilon, args.alpha, args.mirror)
                black = Agent('Black', chess.BLACK)
                black.set_params(random.choice(best_models), args.max_depth, args.n_simulations, args.cputc,
                                 args.epsilon, args.alpha, args.mirror)
                inqueue.put((white, black, args.tau, args.max_turn, args.mirror, False))

            # best 모델들 간의 플레이 데이터를 replay memory에 저장함
            for _ in trange(args.n_selfplay, desc='Selfplay'):
                score, turn, episode = outqueue.get()
                memory.add_samples(episode)

            logger.debug('Selfplay: {} games, {} sec.'.format(args.n_selfplay, time.perf_counter() - stime))
            logger.info('Replay memory: {:.3f} filled {} / {} '.format(
                memory.size / args.max_memory_size, memory.size, args.max_memory_size))

            if memory.size >= args.min_memory_size:
                # replay memory에 충분히 학습 데이터가 모이면 학습을 시작함
                record.xs.append(record.iteration)

                logger.info('Start training')
                # current 모델을 학습모드로 바꿈
                current_model.train()
                if args.cuda:
                    current_model.cuda()

                train_policy_losses = list()
                train_value_losses = list()
                validation_policy_losses = list()
                validation_value_losses = list()
                
                # 미니배치 생성: replay memory에서 학습 데이터와 검증 데이터를 가져옴
                train_set, validation_set = memory.get_dataset(
                    args.n_batches * args.batch_size, args.n_batches * args.batch_size)

                # 학습 알고리즘의 학습률과 모멘텀 갱신
                for param_group in optimizer.param_groups:
                    record.learning_rates.append(param_group['lr'])
                    record.momentums.append(param_group['momentum'])
                record.batch_sizes.append(args.batch_size)

                # n_train 번 학습 수행
                ob, s, m, lm, pi, r, done = train_set
                for it in trange(args.n_train, desc='Training'):
                    for ib in range(args.n_batches):
                        buff.observations[:] = ob[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.states[:] = s[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.moves[:] = m[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.legal_moves[:] = lm[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.pi_prob[:] = pi[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.values[:] = r[args.batch_size * ib: args.batch_size * (ib+1)]
                        buff.dones[:] = done[args.batch_size * ib: args.batch_size * (ib+1)]

                        _ob = Variable(buff.observations)
                        _s = Variable(buff.states)
                        _m = Variable(buff.moves)
                        _lm = Variable(buff.legal_moves)
                        _pi = Variable(buff.pi_prob)
                        _z = Variable(buff.values)

                        _logits, _value_pred = current_model(_ob, _s)

                        if args.mask_legal_moves:
                            # 실제 가능한 수에 마스킹을 함
                            # : 실제로 가능한 수를 제외한 나머지 값은 학습할 필요가 없으므로 0으로 만듬
                            _logits = ((1 - _lm) * -100) + (_lm * _logits)

                        # 신경망 가중치 갱신
                        optimizer.zero_grad()
                        _log_prob = F.log_softmax(_logits, dim=1)
                        policy_loss = -(_pi * _log_prob).mean()
                        value_loss = (_z - _value_pred).pow(2).mean()
                        loss = policy_loss + args.value_loss_coef * value_loss
                        loss.backward()
                        nn.utils.clip_grad_norm(current_model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        train_policy_losses.append(policy_loss.data[0])
                        train_value_losses.append(value_loss.data[0])

                # 학습데이터와 별도로 검증데이터를 이용해 오차를 계산함
                ob, s, m, lm, pi, r, done = validation_set
                for ib in range(args.n_batches):
                    buff.observations[:] = ob[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.states[:] = s[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.moves[:] = m[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.legal_moves[:] = lm[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.pi_prob[:] = pi[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.values[:] = r[args.batch_size * ib: args.batch_size * (ib+1)]
                    buff.dones[:] = done[args.batch_size * ib: args.batch_size * (ib+1)]

                    _ob = Variable(buff.observations, volatile=True)
                    _s = Variable(buff.states, volatile=True)
                    _m = Variable(buff.moves, volatile=True)
                    _lm = Variable(buff.legal_moves, volatile=True)
                    _pi = Variable(buff.pi_prob, volatile=True)
                    _z = Variable(buff.values, volatile=True)

                    _logits, _value_pred = current_model(_ob, _s)

                    if args.mask_legal_moves:
                        _logits = ((1 - _lm) * -100) + (_lm * _logits)  # mask legal moves

                    if ib == 0:
                        _prob = F.softmax(((1 - _lm) * -100) + (_lm * _logits), dim=1)
                        buff.policy_outs[0][:] = _prob.data.cpu()
                        buff.policy_outs[1][:] = _pi.data.cpu()

                    _log_prob = F.log_softmax(_logits, dim=1)
                    policy_loss = -(_pi * _log_prob).mean()
                    value_loss = (_z - _value_pred).pow(2).mean()
                    validation_policy_losses.append(policy_loss.data[0])
                    validation_value_losses.append(value_loss.data[0])

                record.train_policy_losses.append(np.mean(train_policy_losses))
                record.train_value_losses.append(np.mean(train_value_losses))
                record.validation_policy_losses.append(np.mean(validation_policy_losses))
                record.validation_value_losses.append(np.mean(validation_value_losses))
                total_loss = np.mean(validation_policy_losses) + args.value_loss_coef * np.mean(validation_value_losses)
                logger.info('Train      >> P-loss: {:.6f} V-loss: {:.6f}'.format(
                    record.train_policy_losses[-1], record.train_value_losses[-1]))
                logger.info('Validation >> P-loss: {:.6f} V-loss: {:.6f}'.format(
                    record.validation_policy_losses[-1], record.validation_value_losses[-1]))

                # MCTS의 출력과 신경망 Policy 예측의 비교
                viz.policy_out_win = viz.images(
                    tensor=buff.policy_outs,
                    win=viz.policy_out_win if hasattr(viz, 'policy_out_win') else None,
                    nrow=2, opts=dict(title=args.date+': probs and pi', width=910, height=250))
                # Policy Net의 학습오차와 검증오차를 표시
                viz.policy_loss_win = viz.line(
                    X=np.column_stack([record.xs, record.xs]),
                    Y=np.column_stack([record.train_policy_losses, record.validation_policy_losses]),
                    win=viz.policy_loss_win if hasattr(viz, 'policy_loss_win') else None,
                    opts=dict(title=args.date+'<br>policy-loss: {:6f} / {:6f}'.format(
                        record.train_policy_losses[-1], record.validation_policy_losses[-1]), width=450, height=250))
                # Value Net의 학습오차와 검증오차를 표시
                viz.value_loss_win = viz.line(
                    X=np.column_stack([record.xs, record.xs]),
                    Y=np.column_stack([record.train_value_losses, record.validation_value_losses]),
                    win=viz.value_loss_win  if hasattr(viz, 'value_loss_win') else None,
                    opts=dict(title=args.date+'<br>value-loss: {:6f} / {:6f}'.format(
                        record.train_value_losses[-1], record.validation_value_losses[-1]), width=450, height=250))
                # 학습률 변화를 표시
                viz.learning_rate_win = viz.line(
                    X=np.array(record.xs), Y=np.array(record.learning_rates),
                    win=viz.learning_rate_win  if hasattr(viz, 'learning_rate_win') else None,
                    opts=dict(title=args.date+'<br>learning-rate: {:6f}'.format(
                        record.learning_rates[-1]), width=450, height=250))

                if record.iteration % args.eval_interval == 0:
                    # 현재 학습한 모델 성능 평가
                    logger.info('Start evaluation')
                    logger.info('Eval: Current vs. Best')
                    # best 모델들과 current 모델을 평가모드로 바꾸고, 공유 메모리에 복사
                    for best_model in best_models:
                        best_model.cpu().share_memory()
                        best_model.eval()
                    current_model.cpu().share_memory()
                    current_model.eval()

                    # current 모델과 무작위로 선택된 best 모델로 게임을 진행
                    stime = time.perf_counter()
                    for _id in range(args.n_eval):
                        if _id < args.n_eval // 2:
                            white_agent = Agent('Current', chess.WHITE)
                            white_agent.set_params(
                                current_model, args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            black_agent = Agent('Best', chess.BLACK)
                            black_agent.set_params(
                                random.choice(best_models), args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            reversed_reward = False
                        else:
                            white_agent = Agent('Best', chess.WHITE)
                            white_agent.set_params(
                                random.choice(best_models), args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            black_agent = Agent('Current', chess.BLACK)
                            black_agent.set_params(
                                current_model, args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            reversed_reward = True
                        inqueue.put((white_agent, black_agent, 0, args.max_turn, args.mirror, reversed_reward))  # tau=0

                    scores = list()
                    for _ in trange(args.n_eval, desc='Eval: Best'):
                        score, _, _ = outqueue.get()
                        scores.append(score)
                    logger.info('Eval: {} games, {} sec.'.format(args.n_eval, time.perf_counter() - stime))

                    scores = np.array(scores)
                    mean_score = scores.mean(axis=0)
                    logger.info('Mean score: Current: {}, Best: {}'.format(mean_score[0], mean_score[1]))
                    record.eval_scores.append((record.iteration, mean_score[0]))

                    # current 모델 저장
                    torch.save(current_model.state_dict(), '{}/current.pt'.format(args.out_dir))
                    # current 모델과 best 모델을 비교
                    if mean_score[0] > mean_score[1]:
                        pvalue = stats.ttest_rel(scores[:, 0], scores[:, 1]).pvalue
                        logger.info('p-value: {}'.format(pvalue))
                        if pvalue < args.p_value:
                            # current 모델의 성적이 best 모델들의 성적보다 좋고, 그것이 통계적으로 유의미하다면
                            # 가장 오래된 best 모델을 하나 제거하고, current 모델을 best 모델에 복사함
                            for bm_idx in range(args.n_bests-1):
                                best_models[bm_idx].load_state_dict(best_models[bm_idx+1].state_dict())
                            best_models[-1].load_state_dict(current_model.state_dict())
                            shutil.copy('{}/best.pt'.format(args.out_dir),
                                        '{}/best-{}.pt'.format(args.out_dir, record.best_model_version))
                            torch.save(best_models[-1].state_dict(), '{}/best.pt'.format(args.out_dir))
                            record.best_model_version += 1
                            logger.info('New best model: v.{}'.format(record.best_model_version))
                    record.best_model_versions.append((record.iteration, record.best_model_version))

                    # Two Step Search AI와 Stockfish를 대상으로 성능평가
                    # current 모델이 상대방을 이길 때마다 상대방의 level을 높이며 성능을 평가함
                    opponent_cls, opponent_name, opponent_level = opponents[opponent_idx]
                    logger.info('Eval: Current vs. {}'.format(opponent_name))
                    current_model.cpu().share_memory()
                    current_model.eval()

                    stime = time.perf_counter()
                    for _id in range(args.n_eval):
                        if _id < args.n_eval // 2:
                            white_agent = Agent('Current', chess.WHITE)
                            white_agent.set_params(
                                current_model, args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            black_agent = opponent_cls(opponent_name, chess.BLACK)
                            black_agent.level = opponent_level
                            reversed_reward = False
                        else:
                            white_agent = opponent_cls(opponent_name, chess.WHITE)
                            white_agent.level = opponent_level
                            black_agent = Agent('Current', chess.BLACK)
                            black_agent.set_params(
                                current_model, args.max_depth, args.n_simulations, args.cputc,
                                -1., 1., args.mirror)
                            reversed_reward = True
                        inqueue.put((white_agent, black_agent, 0, args.max_turn, args.mirror, reversed_reward))  # tau=0

                    scores = list()
                    for _ in trange(args.n_eval, desc='Eval: {}'.format(opponent_name)):
                        score, turn, episode = outqueue.get()
                        scores.append(score)
                    logger.info('Eval: {} games, {} sec.'.format(args.n_eval, time.perf_counter() - stime))

                    scores = np.array(scores)
                    mean_score = scores.mean(axis=0)
                    logger.info('Mean score: Current: {}, {}: {}'.format(mean_score[0], opponent_name, mean_score[1]))
                    record.opponent_level.append((record.iteration, opponent_idx))

                    # current 모델이 좋은 성능을 보이면, 상대방의 level을 높이고,
                    # 낮은 성능을 보이면 상대방의 level을 낮춤
                    if mean_score[0] > 0.55:
                        next_opponent_idx = min(len(opponents) - 1, opponent_idx + 1)
                        logger.info('Change opponent: {} --> {}'.format(
                            opponents[opponent_idx][1], opponents[next_opponent_idx][1]))
                        opponent_idx = next_opponent_idx
                    elif mean_score[0] < 0.45:
                        next_opponent_idx = max(0, opponent_idx - 1)
                        logger.info('Change opponent: {} --> {}'.format(
                            opponents[opponent_idx][1], opponents[next_opponent_idx][1]))
                        opponent_idx = next_opponent_idx

                    # best model이 갱신된 횟수 갱신
                    xs, best_model_versions = zip(*record.best_model_versions)
                    viz.version_win = viz.line(
                        X=np.array(xs), Y=np.array(best_model_versions),
                        win=viz.version_win if hasattr(viz, 'version_win') else None,
                        opts=dict(title=args.date+': version: {}'.format(record.best_model_versions[-1]),
                                  width=450, height=250))
                    # current vs. bset 에서 current 모델의 승률
                    xs, scores = zip(*record.eval_scores)
                    viz.score_win = viz.line(
                        X=np.array(xs), Y=np.array(scores),
                        win=viz.score_win if hasattr(viz, 'score_win') else None,
                        opts=dict(title=args.date+': score: {}'.format(record.eval_scores[-1]),
                                  width=450, height=250))
                    # 상대방 level 표시
                    xs, opponent_levels = zip(*record.opponent_level)
                    viz.opponent_level_win = viz.line(
                        X=np.array(xs), Y=np.array(opponent_levels),
                        win=viz.opponent_level_win if hasattr(viz, 'opponent_level_win') else None,
                        opts=dict(title=args.date+': opponent level: {}'.format(record.opponent_level[-1]),
                                  width=450, height=250))
                    # 가능하다면 gnuplot으로 console 창에 상대방 level 표시
                    ascii_plot(xs, opponent_levels, title='opponent_levels', print_out=True)

                # learning rate 갱신
                if record.iteration in milestones:
                    args.learning_rate = lr_degration * args.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate
                    param_group['momentum'] = args.momentum

                if args.batch_size != buff.batch_size:
                    buff = Buffer(args)

                record.iteration += 1
                record.wall_times.append(int(time.perf_counter() - record.main_start_time))

            gc.collect()
            logger.info('Loop: {:.1f} sec. elapsed'.format(time.perf_counter() - loop_start_time))

            while len(CMD_QUEUE) > 0:
                cmd = CMD_QUEUE.pop(0)
                cmd, *arguments = cmd.split(' ')
                if hasattr(Operators, cmd):
                    func = getattr(Operators, cmd)

                    try:
                        args, record, current_model, best_models = func(args, record, current_model, best_models, arguments)
                    except Exception as exc:
                        logger.error('command execution failure: {} -> {}'.format(cmd, exc))
                else:
                    logger.warning('Worng command: {}'.format(cmd))
    finally:
        [worker.terminate() for worker in workers]
        inqueue.close()
        outqueue.close()


if __name__ == '__main__':

    train()