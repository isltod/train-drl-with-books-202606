from continuous_observ_space import *
import gymnasium as gym
from tqdm import tqdm

# Mountain Car 환경 생성, 난수 시드 지정...
env = gym.make("MountainCar-v0", render_mode="rgb_array")
seed_everything(env)

print(f"Observation Space: {env.observation_space}")
print(f"Low: {env.observation_space.low}")
print(f"High: {env.observation_space.high}")


# 직접 안 만들고 ObservationWrapper 상속받아 처리...
class StateAggregationEnv(gym.ObservationWrapper):
    def __init__(self, env, bins, low, high):
        super().__init__(env)
        self.bins = bins
        self.low = low
        self.high = high
        # 이산화된 상태 공간 정의 (MultiDiscrete)
        self.observation_space = gym.spaces.MultiDiscrete(bins)

        # 각 차원별로 구간(Bucket) 생성 - 구간 경계를 리스트로 나열
        self.buckets = [np.linspace(l, h, b - 1) for l, h, b in zip(low, high, bins)]

    def observation(self, obs):
        # 연속적인 관측값을 해당 버킷의 인덱스로 변환(l <= x < h로 인덱스 찾기)
        indices = tuple(np.digitize(x, b) for x, b in zip(obs, self.buckets))
        return indices


# 래퍼 적용 (20x20 그리드)
bins = np.array([20, 20])
low = env.observation_space.low
high = env.observation_space.high
saenv = StateAggregationEnv(env, bins=bins, low=low, high=high)

print(f"Modified observation space: {saenv.observation_space}")
print(f"Sample state (discrete): {saenv.observation_space.sample()}")

# Q-테이블 초기화
action_values = np.zeros((20, 20, 3))


# Epsilon-Greedy 정책
def policy(state, epsilon=0.0):
    if np.random.random() < epsilon:
        # 0 왼쪽 가속, 1 정지, 2 오른쪽 가속
        return np.random.randint(3)
    else:
        # 상태에 해당하는 Q값들 가져오기
        av = action_values[state]
        # 최대 Q값을 가진 행동 선택 (동점시 랜덤)
        return np.random.choice(np.flatnonzero(av == av.max()))


# SARSA 알고리즘으로 학습...
def sarsa(env, action_values, policy, episodes, alpha=0.1, gamma=0.99, epsilon=0.2):
    stats = {"Returns": []}

    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        action = policy(state, epsilon)
        done = False
        total_return = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = policy(next_state, epsilon)

            # SARSA 업데이트
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            q_val = action_values[state][action]
            next_q_val = action_values[next_state][next_action]
            target = reward + gamma * next_q_val
            action_values[state][action] += alpha * (target - q_val)

            state = next_state
            action = next_action
            total_return += reward

        stats["Returns"].append(total_return)

    return stats


# 학습 실행 (20,000 에피소드)
print("Training with State Aggregation...")
stats = sarsa(saenv, action_values, policy, episodes=20000, alpha=0.1, epsilon=0.1)
# 학습 결과 그래프...
plot_stats(stats)
# 어떤 상태를 좋고 나쁘게 평가하는지...
plot_tabular_cost_to_go(action_values, xlabel="Car Position", ylabel="Velocity")
