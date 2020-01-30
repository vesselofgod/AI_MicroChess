
"""
OBSERVATION_SHAPE:
    12 channels: 
        말 종류마다 하나의 채널을 사용 함(P, R, N, B, Q, K, p, r, n, b, q, k)
        백은 대문자, 흑은 소문자로 표시함
    5 rows:
        세로
    4 columns:
        가로
    
    특정 위치에 기물이 위치하면 1., 기물이 없으면 0.으로 표시
LEN_STATE
    지금이 백의 차례인가?, 흑의 차례인가? , 
    현재 플레이어에게 castling 권리가 있는가?, 다음 플레이어에게 castling 권리가 있는가?
    정보를 1., 과 0.로 표시
N_ACTIONS
    가능한 모든 행동의 개수, 실제 기물의 존재 여부와 관계없이 
    가능한 모든 출발 지점 20개와 목표 지점 20개를 곱한 400개를 가능한 모든 행동으로 함 
"""
OBSERVATION_SHAPE = (12, 5, 4)
LEN_STATE = 4
N_ACTIONS = (5 * 4) * (5 * 4)