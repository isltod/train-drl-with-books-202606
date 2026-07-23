import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("CartPole-v1", render_mode="rgb_array")

# 상태값은 (카트 위치, 카트 속도, 막대 각도, 막대 각속도) 4차원
# 상태 범위 확인 - 위치 -4.8~4.8, 속도 inf, 각도 -0.42~0.42(-24~24도), 각속도 inf...
print("Observation Space High:", env.observation_space.high)
print("Observation Space Low:", env.observation_space.low)

# 이산화를 위한 임의의 경계 설정
# 속도, 각속도는 무한대지만, 유의미한 범위 내로 자른다.
# 원래 코드는 이상해서 바꿔도 봤는데...
upper_bounds = [2.4, 0.5, 0.418, 0.87]  # [m, m/s, rad, rad/s]
lower_bounds = [-2.4, -0.5, -0.419, -0.87]
# 이렇게...그래도 학습이 잘 되질 않는다..왜일까...
upper_bounds = [4.8, 0.5, 0.418, 0.87]  # [m, m/s, rad, rad/s]
lower_bounds = [-4.8, -0.5, -0.419, -0.87]

cartpole_bounds = list(zip(lower_bounds, upper_bounds))
print("설정된 경계값:", cartpole_bounds)

# 4차원 버킷 설정: 위치(1), 속도(1), 각도(6), 각속도(12)
# 중요하지 않은 차원의 버킷 수를 줄여서 크기를 최적화...위치, 속도는 하나도 안 중요하다는 얘기네...
# n_buckets = (1, 1, 6, 12)
n_buckets = (16, 16, 96, 192)

# Q-테이블 초기화 (1x1x6x12 x 2) - 마지막에 행동 공간 차원을 튜플로 붙여야 전체가 한 배열로 정리...
q_table = np.zeros(n_buckets + (env.action_space.n,))


def discretize_cartpole(state, bounds, n_buckets):
    discretized = []
    for i in range(len(state)):
        # 범위 밖의 값은 클리핑 - 큰 값은 min으로 자르고 다시 작은 값은 max로 자른다...
        val = max(bounds[i][0], min(state[i], bounds[i][1]))

        # (val - low) / (high - low) = 0~1 현재 상태값을 범위 내 비율로 표시하고
        scaling = (val - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        # 그 비율로 이산화된 인덱스 선택...
        new_obs = int(round((n_buckets[i] - 1) * scaling))
        # 그걸 다시 인덱스 범위 밖으로 나가지 않게 자르기? 왜 이렇게 두 번 하지?
        new_obs = min(n_buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)


# 하이퍼파라미터
EPISODES = 100000
ALPHA = 0.1
GAMMA = 0.99
# 일단 시작은 무조건 확률적으로 결정한다...
EPSILON = 1.0
DECAY_RATE = 0.005  # 빠르게 감소

scores = []

# 주어진 에피소드 수만큼 반복하면서...
for episode in tqdm(range(EPISODES)):
    # 첫 번째 원래 연속 상태 받아서 이산 상태로 변경하고 시작
    current_state_continuous, _ = env.reset()
    current_state = discretize_cartpole(
        current_state_continuous, cartpole_bounds, n_buckets
    )

    done = False
    score = 0

    # 하나의 에피소드 내에서 반복해서 Q 테이블 갱신...
    while not done:
        # 1. 선택은 ε-greedy로...처음에는 ε 100%로 시작해서 에피소드마다 줄여가며 전략 수정...
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        # 2. 다음 상태를 연속값으로 받고, 그걸 이산화하고...
        next_state_continuous, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_cartpole(
            next_state_continuous, cartpole_bounds, n_buckets
        )

        # 다음 상태의 maxQ와 현재 Q
        best_next_q = np.max(q_table[next_state])
        current_q = q_table[current_state + (action,)]

        # 3. Q-Learning 업데이트
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',:)) - Q(s,a))
        new_q = current_q + ALPHA * (reward + GAMMA * best_next_q - current_q)
        q_table[current_state + (action,)] = new_q

        # 다음 상태를 현재 상태로 놓고 다음 에피소드 반복...
        current_state = next_state
        # 점수는 즉각 반환을 그냥 단순 합산한 거...
        score += reward

    # 에피소드 끝나면 점수 기록하고 엡실론 감쇠...
    scores.append(score)
    EPSILON = max(0.01, EPSILON * np.exp(-DECAY_RATE))

    if episode % 10000 == 0:
        print(f"Episode: {episode}, Score: {score}, Epsilon: {EPSILON:.2f}")

print("CartPole 학습 완료!")
env.close()

# 결과 시각화
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Duration (Score)")
plt.title("CartPole Training Progress")
plt.show()
