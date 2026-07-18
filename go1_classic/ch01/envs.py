import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Maze(gym.Env):
    def __init__(self):
        super().__init__()
        # [0~4, 0~4] 중에 한 쌍 반환
        self.observation_space = spaces.MultiDiscrete([5, 5])  # 5x5 그리드
        # 0~3 중에 한 숫자 반환
        self.action_space = spaces.Discrete(4)  # 상, 우, 하, 좌
        self.state = np.array([0, 0])
        self.target = np.array([4, 4])
        # 그냥 텍스트로 이렇게 선언하면 알아서 그려주나? 그럴리가...
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0])
        return self.state, {}

    def step(self, action):
        # 0: Up, 1: Right, 2: Down, 3: Left (원본 노트북 기준)
        # 좌표계: (row, col) -> (y, x)로 가정.
        # 행동별 이동을 만들어두고 현재 상태에서 move로 다음 상태 계산
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        move = moves[action]
        next_state = self.state + np.array(move)

        # 그리드 밖으로 나가지 않도록 클리핑 (0~4 사이)
        next_state = np.clip(next_state, 0, 4)
        self.state = next_state

        # 목표 도달 여부 확인
        terminated = np.array_equal(self.state, self.target)
        truncated = False  # 시간 제한 없음
        reward = -1.0  # 매 스텝마다 -1 보상

        return self.state, reward, terminated, truncated, {}

    def render(self):
        # 시각화를 위한 간단한 그리드 이미지 생성
        grid = np.zeros((5, 5, 3), dtype=np.uint8) + 255  # 흰색 배경
        # 현재 위치 (파란색)
        grid[self.state[0], self.state[1]] = [0, 0, 255]
        # 목표 위치 (녹색)
        grid[self.target[0], self.target[1]] = [0, 255, 0]

        # 이미지를 크게 보기 위해 확대
        # - 크로네커 곱을 하는데, grid 행렬의 각 원소를 뒤 행렬에 곱해서 원래 위치에 블록 행렬로 나열 - 뻥튀기 된다...
        grid_large = np.kron(grid, np.ones((40, 40, 1), dtype=np.uint8))
        return grid_large
