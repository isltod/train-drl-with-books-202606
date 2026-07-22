import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# 환경 생성 (시각화 모드 설정)
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# 위치 -1.2~0.6, 속도 -0.07~0.07
print("상태 공간(State Space):", env.observation_space)
print("  - Low:", env.observation_space.low)
print("  - High:", env.observation_space.high)
# 왼쪽 가속 0, 정지 1, 오른쪽 가속 2
print("행동 공간(Action Space):", env.action_space)

# 초기화 테스트
state, _ = env.reset()
print("초기 상태:", state)

# 1. 상태를 이산화할 버킷(구간)의 개수 설정
n_buckets = (20, 20)  # 위치 20개, 속도 20개 구간

# 2. 각 차원의 범위(Range) 계산
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
# 속도 범위가 너무 크거나 작을 수 있으므로 적절히 조정할 수도 있으나, 여기선 기본값 사용
# MountainCar의 속도 범위는 보통 [-0.07, 0.07] 정도임


def discretize(state, bounds, n_buckets):
    """
    연속적인 상태(state)를 이산적인 상태 인덱스(tuple)로 변환한다.
    """
    discretized = []
    for i in range(len(state)):
        # 현재 값의 비율 계산 (x - low) / (high - low) = (0.0 ~ 1.0)...즉 최저에서 최대 사이를 몇 %
        scaling = (state[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])

        # 버킷 개수에 맞게 인덱스로 변환, 0에서 시작하는 인덱스로 해당 %에 해당하는 인덱스 선택
        new_obs = int(round((n_buckets[i] - 1) * scaling))

        # 인덱스가 범위를 벗어나지 않도록 클리핑 - max로 음수 제거, min으로 19 넘는 값 제거
        new_obs = min(n_buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)

    return tuple(discretized)


# 테스트
test_state = np.array([-0.5, 0.0])  # 대략 중간 위치, 정지 상태
discrete_state = discretize(test_state, state_bounds, n_buckets)
print(f"연속 상태 {test_state} -> 이산 상태 {discrete_state}")

# Q Learning
# Q-테이블 초기화 (이산화 한 상태 20x20 x 행동 3)
# 행동 3을 그대로 더하면 [20,20] + 3 = [23,23] 되는데, 뒤를 튜플로 묶어서 더하면 [20,20] + (3,) = [20,20,3]이 된다...
q_table = np.zeros(n_buckets + (env.action_space.n,))
# 하이퍼파라미터 설정
EPISODES = 5000  # 총 에피소드 수
ALPHA = 0.1  # 학습률
GAMMA = 0.99  # 할인율
EPSILON = 1.0  # 초기 탐험 확률
MIN_EPSILON = 0.01  # 최소 탐험 확률
DECAY_RATE = 0.001  # 탐험 감소율 (Exponential Decay) or Linear
# 학습 루프
scores = []

# 주어진 에피소드 수만큼 돌면서...
for episode in tqdm(range(EPISODES)):
    # 환경 초기화 및 상태 이산화
    current_state_continuous, _ = env.reset()
    current_state = discretize(current_state_continuous, state_bounds, n_buckets)

    done = False
    score = 0

    # 한 에피소드 돌면서
    while not done:
        # 1. 행동 선택 (Epsilon-Greedy)
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[current_state])

        # 2. 행동 수행
        next_state_continuous, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 상태는 그때마다 이산화해서 사용한다...
        next_state = discretize(next_state_continuous, state_bounds, n_buckets)

        # 다음 상태의 maxQ
        best_next_q = np.max(q_table[next_state])
        # 현재 Q - 마찬가지로 그냥 더하면 [x1, v1] + a1 = [x1+a1, v1+a1]인데, 튜플로 더하면 [x1, v1, a1]이 된다...
        current_q = q_table[current_state + (action,)]

        # 3. Q-Learning 업데이트
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',:)) - Q(s,a))
        new_q = current_q + ALPHA * (reward + GAMMA * best_next_q - current_q)
        q_table[current_state + (action,)] = new_q

        # 다음 이산 상태를 현재 이산 상태로 놓고 반복...
        # 연속인 원래 상태는 더 이상 필요 없고, 행동에 따른 다음 상태만 다시 조회하면 되니까...
        current_state = next_state
        score += reward

    scores.append(score)

    # 한 에피소드 끝나면 엡실론 감쇠
    EPSILON = max(MIN_EPSILON, EPSILON * np.exp(-DECAY_RATE))

    if episode % 500 == 0:
        print(f"Episode: {episode}, Score: {score}, Epsilon: {EPSILON:.2f}")

print("Mountain Car 학습 완료!")
env.close()

# 학습된 Q 테이블로 에이전트가 언덕을 올라가는지 테스트...
# 다시 환경 만들고
env = gym.make("MountainCar-v0", render_mode="rgb_array")
state_continuous, _ = env.reset()
state = discretize(state_continuous, state_bounds, n_buckets)
done = False

img = plt.imshow(env.render())
plt.axis("off")
# 연속 그래프 모드 켜기
plt.ion()

while not done:
    # 학습된 정책(Greedy) 사용
    action = np.argmax(q_table[state])

    next_state_continuous, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    state = discretize(next_state_continuous, state_bounds, n_buckets)

    img.set_data(env.render())
    # 요게 연속으로 그리는데...
    plt.draw()
    plt.pause(0.05)

# 연속 그래프 모드 끄기
plt.ioff()
env.close()
