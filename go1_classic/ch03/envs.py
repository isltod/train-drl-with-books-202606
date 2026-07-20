import gymnasium as gym
import numpy as np
from gymnasium import spaces


# 이건 ch02와 같고...
class Maze(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(25)  # 5x5 그리드
        self.action_space = spaces.Discrete(4)  # 상, 우, 하, 좌
        self.state = 0
        self.target = 24  # 목표 지점 (4, 4)
        self.row_len = 5
        self.render_mode = "rgb_array"

        # 전이 모델 P[state][action] = [(prob, next_state, reward, done)] 계산
        self.P = {}
        for s in range(25):
            self.P[s] = {}
            for a in range(4):
                self.P[s][a] = []
                row, col = divmod(s, self.row_len)

                # 행동에 따른 좌표 이동
                moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
                dr, dc = moves[a]
                next_row, next_col = row + dr, col + dc

                # 맵 경계 처리
                next_row = np.clip(next_row, 0, 4)
                next_col = np.clip(next_col, 0, 4)
                next_state = next_row * self.row_len + next_col

                # 보상 및 종료 조건 설정
                done = next_state == self.target
                reward = -1.0
                if s == self.target:  # 목표 상태에서는 이동 불가
                    next_state = self.target
                    reward = 0.0
                    done = True

                self.P[s][a].append((1.0, next_state, reward, done))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        prob, next_state, reward, done = self.P[self.state][action][0]
        self.state = next_state
        return self.state, reward, done, False, {}

    def render(self):
        grid = np.zeros((5, 5, 3), dtype=np.uint8) + 255
        curr_r, curr_c = divmod(self.state, 5)
        target_r, target_c = divmod(self.target, 5)
        grid[curr_r, curr_c] = [0, 0, 255]  # 에이전트 (파란색)
        grid[target_r, target_c] = [0, 255, 0]  # 목표 (녹색)
        return np.kron(grid, np.ones((40, 40, 1), dtype=np.uint8))
