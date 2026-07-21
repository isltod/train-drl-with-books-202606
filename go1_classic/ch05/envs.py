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
        # 5x5 그리드, 상태는 (row, col) 좌표로 표현 (MultiDiscrete)
        self.observation_space = spaces.MultiDiscrete([5, 5])
        # 행동: 0:Up, 1:Right, 2:Down, 3:Left
        self.action_space = spaces.Discrete(4)
        self.state = np.array([0, 0])
        self.target = np.array([4, 4])
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0])
        return self.state, {}

    def step(self, action):
        # 상, 우, 하, 좌 이동
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        move = moves[action]

        # 이동 후 좌표 (맵 밖으로 나가지 않도록 clip)
        next_state = self.state + np.array(move)
        next_state = np.clip(next_state, 0, 4)
        self.state = next_state

        # 종료 조건 및 보상
        terminated = np.array_equal(self.state, self.target)
        reward = -1.0  # 매 스텝 비용 발생
        if terminated:
            reward = 0.0  # 목표 도달 시 추가 비용 없음

        return self.state, reward, terminated, False, {}

    def render(self):
        # 5x5 그리드 이미지 생성
        grid = np.zeros((5, 5, 3), dtype=np.uint8) + 255
        grid[self.state[0], self.state[1]] = [0, 0, 255]  # 파란색: 에이전트
        grid[self.target[0], self.target[1]] = [0, 255, 0]  # 녹색: 목표
        return np.kron(grid, np.ones((40, 40, 1), dtype=np.uint8))


# ---------------------------------------------------------
# 2. 시각화 및 테스트를 위한 헬퍼 함수 정의
# ---------------------------------------------------------
def plot_action_values(action_values):
    # 각 상태에서 가장 높은 Q값(가치)을 히트맵으로 표시
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


def plot_policy(action_values):
    # 각 상태에서 가장 Q값이 높은 행동을 화살표로 표시
    best_actions = np.argmax(action_values, axis=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros((5, 5)), cmap="gray", vmin=0, vmax=1)  # 배경

    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}  # dx, dy
    for r in range(5):
        for c in range(5):
            if r == 4 and c == 4:
                continue  # 목표 지점 제외
            action = best_actions[r, c]
            dx, dy = arrows[action]
            plt.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc="red", ec="red")
    plt.grid(color="white")
    plt.title("Optimal Policy (Greedy)")
    plt.show()


def test_agent(env, policy_fn, action_values):
    state, _ = env.reset()
    done = False
    step = 0
    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("Test Run")
    # 연속 그래프 모드 켜기
    plt.ion()

    while not done and step < 20:
        # 테스트 시에는 탐험 없이 가장 좋은 행동(Greedy) 선택
        action = policy_fn(tuple(state), epsilon=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        img.set_data(env.render())
        # 요게 연속으로 그리는데...
        plt.draw()
        plt.pause(0.2)
        step += 1
    # 연속 그래프 모드 끄기
    plt.ioff()
    print("테스트 종료")
