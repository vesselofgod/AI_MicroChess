# -*- coding: utf-8 -*-
"""
argument 처리부분
"""
__author__ = 'Hyunsoo Park, Game AI Team, NCSOFT'


import argparse
from datetime import datetime
import os
import tempfile

import torch
import torch.multiprocessing as mp

from agents.self_learning.utils import get_logger


logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8097, help='visdom port')
    parser.add_argument('-o', '--out-dir', type=str,
                        help='출력 파일을 저장할 경로, 기본값이 지정되지 않으면 '
                             './out/{args.arch}/{args.date} 에 저장함')
    # NN
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA 사용, CUDA를 사용한 경우는 성능이 충분히 검증되지 않았음')
    parser.add_argument('--arch', type=str, default='C64r6L1024r',
                        help='인공 신경망 구조 정의, models.py 에 있는 torch.nn 클래스')
    parser.add_argument('--batch-size', type=int, default=32, help='학습 데이터 배치크기')
    parser.add_argument('--n-batches', type=int, default=256,
                        help='미니배치안에 포함될 배치 개수, '
                             '미니배치 크기 = args.batch_size * args.n_batches')
    parser.add_argument('--memory-path', type=str,
                        help='Replay memory 를 저장할 경로, 기본값이 지정되지 않으면 임시폴더 사용함')
    parser.add_argument('--max-memory-size', type=int, default=640000,
                        help='Replay memory 최대 크기')
    parser.add_argument('--min-memory-size', type=int, default=160000,
                        help='학습을 시작하는 최소 데이터 샘플 개수, '
                             'replay memory 를 채우는데 너무 오래 걸리기 대문에 사용')
    parser.add_argument('-LR', '--learning-rate', type=float, default=0.01, help='SGD 파라미터, 학습률')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 파라미터, 모멘텀')
    parser.add_argument('--l2-penalty', type=float, default=0.0001, help='SGD 파라미터, L2-norm')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Gradient 값이 너무 클 경우 이 값으로 clamping')
    parser.add_argument('--value-loss-coef', type=float, default=0.02,
                        help='value-loss 가중치, 전체 loss 에서 value-loss 비중을 조절 하기 위해 사용')
    # MCTS
    parser.add_argument('--max-depth', type=int, default=32,
                        help='MCTS 파라미터, 최대 탐색 깊이')
    parser.add_argument('--n-simulations', type=int, default=500,
                        help='MCTS 파라미터, 시뮬레이션 횟수')
    parser.add_argument('--cputc', type=float, default=1.0, help='UCB에서 탐색과 활용을 조정하는 상수')
    parser.add_argument('--epsilon', type=float, default=0.2, help='탐색 가중치 epsilon')
    parser.add_argument('--alpha', type=float, default=0.8, help='활용 가중치 alpha')
    parser.add_argument('--tau', type=float, default=10, help='tau')
    # Selfplay
    parser.add_argument('-N', '--n-iter', type=int, default=600, help='총 반복횟수')
    parser.add_argument('--n-workers', type=int, default=-1, help='시뮬레이션 워커 개수, 초기값은 CPU core 개수 - 1')
    parser.add_argument('--n-selfplay', type=int, default=128, help='반복 한번에서 셀프플레이 횟수')
    parser.add_argument('--n-eval', type=int, default=32, help='반복 한번에서 평가 횟수')
    parser.add_argument('--n-train', type=int, default=10, help='반복 한번에서 학습 횟수')
    parser.add_argument('--eval-interval', type=int, default=1, help='평가 간격')
    parser.add_argument('--n-bests', type=int, default=3, help='유지하고 있는 가장 좋은 모델의 개수')
    parser.add_argument('--p-value', type=float, default=0.1,
                        help='가장 좋은 모델과 현재모델의 성능을 비교할 때 사용, '
                             'p-value를 계산하여 현재 모델의 좋음이 통계적으로 유의미 할 때만 모델을 교체함')
    parser.add_argument('--max-turn', type=int, default=80, help='최대 턴')
    parser.add_argument('--mask-legal-moves', action='store_false', default=True, 
                        help='실행 불가능한 수를 제거하는 마스크를 사용할 지 여부')
    parser.add_argument('--mirror', action='store_true', default=True, 
                        help='게임 데이터를 white vs. black 에서 나 vs. 상대방으로 변경함,'
                             '두 배의 데이터를 학습에 사용할 수 있기 때문에 효율적임'
                             '이것을 사용하지 않은 경우에 대해서는 충분한 테스트가 되어 있지 않음')
    parser.add_argument('--seed', type=int, default=0, help='난수 생성기 시드값')
    args = parser.parse_args()

    # 학습 시작시간 설정
    args.date = datetime.today().isoformat('-').replace(':', '-').split('.')[0]
    logger.info('Start time: {}'.format(args.date))

    # 출력 경로 설정
    if args.out_dir is None:
        # 경로가 설정되어 있지 않으면 models/{신경망구조}/{시작시간} 으로 설정됨
        args.out_dir = 'models/{}/{}'.format(args.arch, args.date)
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info('Output path: {}'.format(args.out_dir))

    # cuda 설정
    args.cuda = args.cuda and torch.cuda.is_available()
    logger.info('CUDA enabled' if args.cuda else 'CUDA disabled')

    # replay memory 경로 설정
    if args.memory_path is None:
        args.memory_path = tempfile.mkdtemp()
    os.makedirs(args.memory_path, exist_ok=True)
    logger.info('Replay memory path: {}'.format(args.memory_path))

    # replay memory에 얼마나 데이터가 모이면 학습을 시작할 것인지에 대한 설정 
    if not (0 < args.min_memory_size <= args.max_memory_size):
        args.min_memory_size = args.max_memory_size
    logger.info('Training start when {} data samples collected'.format(args.min_memory_size))

    if args.n_workers < 0:
        args.n_workers = mp.cpu_count() - 1
    logger.info('Run {} workers'.format(args.n_workers))

    return args
