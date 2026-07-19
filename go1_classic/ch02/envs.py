import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Maze(gym.Env):
    def __init__(self):
        super().__init__()
        # 이번에는 0~24: 5x5 그리드 평탄화
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Right, 2:Down, 3:Left
        self.state = 0
        self.target = 24  # (4, 4) 위치
        self.row_len = 5
        self.render_mode = "rgb_array"

        # 가치 반복을 위한 전이 모델 (P) 미리 계산: P[state][action] = [(prob, next_state, reward, done)]
        self.P = {}
        for s in range(25):
            # 전이확률을 P로, 딕셔너리 안에 상태를 키로 다시 딕셔너리
            self.P[s] = {}
            for a in range(4):
                # 행동 별로 리스트 추가 P = {state: {action1: [], action2: []...}
                self.P[s][a] = []
                # 현재 위치 계산 - 0~24를 5로 나눈 몫과 나머지 튜플
                row, col = divmod(s, self.row_len)

                # 행동에 따른 이동은 결정론적으로...그럼 P는 모든 상태에서 1....
                moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
                dr, dc = moves[a]
                next_row, next_col = row + dr, col + dc

                # 맵 밖으로 나가지 않도록 제한
                next_row = np.clip(next_row, 0, 4)
                next_col = np.clip(next_col, 0, 4)
                next_state = next_row * self.row_len + next_col

                # 보상 및 종료 조건
                done = next_state == self.target
                reward = -1.0  # 매 스텝 비용 발생
                if s == self.target:  # 목표 지점에서는 이동 불가 (종료 상태)
                    next_state = self.target
                    reward = 0.0
                    done = True

                # (전이확률 결정론적으로 1.0, 다음 상태, 보상, 종료 여부)
                # 원래는 확률적이면 리스트 안에 이런 튜플이 여러개 있어야 하는데, 이건 결정론적으로 정해지니 한 개만...
                self.P[s][a].append((1.0, next_state, reward, done))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        # 모델 P를 이용해 step 시뮬레이션 - 여긴 P가 deterministic하니 항상 첫 번째 원소를 반환
        prob, next_state, reward, done = self.P[self.state][action][0]
        self.state = next_state
        # truncated는 항상 false
        return self.state, reward, done, False, {}

    def render(self):
        # 5x5 그리드에 RGB 3차원, 초기화는 다 흰색 255값으로..
        grid = np.zeros((5, 5, 3), dtype=np.uint8) + 255
        # 현재 위치와 목표 지점
        curr_r, curr_c = divmod(self.state, 5)
        target_r, target_c = divmod(self.target, 5)
        grid[curr_r, curr_c] = [0, 0, 255]  # 파란색: 에이전트
        grid[target_r, target_c] = [0, 255, 0]  # 녹색: 목표
        # 크로네커 곱으로 40배 확대...
        return np.kron(grid, np.ones((40, 40, 1), dtype=np.uint8))
