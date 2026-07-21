import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------
# 1. 커스텀 Maze 환경 정의 (Gymnasium 기반)
# ---------------------------------------------------------
class Maze(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([5, 5])  # 5x5 그리드
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Right, 2:Down, 3:Left
        self.state = np.array([0, 0])
        self.target = np.array([4, 4])
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0])
        return self.state, {}

    def step(self, action):
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        move = moves[action]
        next_state = np.clip(self.state + np.array(move), 0, 4)
        self.state = next_state

        terminated = np.array_equal(self.state, self.target)
        reward = -1.0
        if terminated:
            reward = 0.0

        return self.state, reward, terminated, False, {}

    def render(self):
        grid = np.zeros((5, 5, 3), dtype=np.uint8) + 255
        grid[self.state[0], self.state[1]] = [0, 0, 255]
        grid[self.target[0], self.target[1]] = [0, 255, 0]
        return np.kron(grid, np.ones((40, 40, 1), dtype=np.uint8))


# 시각화 헬퍼 함수
def plot_action_values(action_values):
    values = np.max(action_values, axis=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(values, cmap="coolwarm", interpolation="none")
    for r in range(5):
        for c in range(5):
            plt.text(
                c, r, f"{values[r, c]:.1f}", ha="center", va="center", color="black"
            )
    plt.colorbar(label="Max Q-value")
    plt.title("Action Value Table (Max Q)")
    plt.show()


def plot_policy(target_policy):
    # Target Policy는 Deterministic(결정적)하므로 바로 argmax 사용
    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros((5, 5)), cmap="gray", vmin=0, vmax=1)
    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
    for r in range(5):
        for c in range(5):
            if r == 4 and c == 4:
                continue
            # target_policy[r, c]는 확률 분포가 아니라 행동 인덱스 자체를 저장한다고 가정
            action = target_policy[r, c]
            dx, dy = arrows[action]
            plt.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc="red", ec="red")
    plt.grid(color="white")
    plt.title("Optimal Target Policy")
    plt.show()
