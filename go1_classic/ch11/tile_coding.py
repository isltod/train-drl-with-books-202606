from continuous_observ_space import *
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from tqdm import tqdm


# 타일 코딩은 경계를 겹치게 잡는 방법이라는데...
class TileCodingEnv(gym.ObservationWrapper):
    def __init__(self, env, bins, low, high, n=4):
        super().__init__(env)
        self.n = n  # 타일링 개수
        self.bins = bins
        self.low = low
        self.high = high
        # state aggregation은 여기서 그냥 Discrete 만들어서 bucket으로 사용했는데...
        # tile coding에서 상태 공간은 (n, bins[0], bins[1]) 형태가 된다고...이게 조금씩 앞 뒤와 겹치는 모양...
        # Gym Space로는 표현하기 복잡해서 그렇다고 정해두기만 하고 observation에서 튜플 리스트를 반환하도록 함

        # 각 차원의 구간 너비 계산
        self.tile_width = (high - low) / bins
        # 각 타일링별 오프셋(Offset) 생성
        self.offsets = []
        for i in range(n):
            # 타일링마다 넓이/n만큼 오른쪽으로 이동시키는 오프셋
            offset = low + i * (self.tile_width / n)
            self.offsets.append(offset)

        # 각 타일링별 버킷 경계 생성
        self.tilings_buckets = []
        for i in range(n):
            # 오프셋만큼 이동된 경계선 생성
            buckets = [
                np.linspace(l, h, b - 1)
                for l, h, b in zip(
                    self.offsets[i], self.offsets[i] + (high - low), bins
                )
            ]
            self.tilings_buckets.append(buckets)

    # 이제보니 이게 그냥 막 함수가 아니고 상위 클래스 함수를 오버라이딩 하는 모양이네...
    # reset 해도 이 함수가 호출되네...step 할 때도 호출되겠지...
    def observation(self, obs):
        # n개의 타일링 각각에 대해 인덱스를 계산
        indices = []
        for i in range(self.n):
            # 그러니까 4개의 타일을 이용해서 그게 각각 어떤 이산 공간(x, v)에 들어가는지 인덱스를 검사해서
            # 그 네 개를 묶어서 반환하나?
            # 범위를 벗어나는 경우를 대비해 bins-1로 클리핑해주는 것이 안전함
            idx = tuple(
                np.clip(np.digitize(x, b), 0, bn - 1)
                for x, b, bn in zip(obs, self.tilings_buckets[i], self.bins)
            )
            indices.append(idx)
        return tuple(indices)


# Mountain Car 환경 생성, 난수 시드 지정...
env = gym.make("MountainCar-v0", render_mode="rgb_array")
seed_everything(env)

# 래퍼 적용 (4개의 타일링 사용)
tilings = 4
bins = np.array([20, 20])
low = env.observation_space.low
high = env.observation_space.high
tcenv = TileCodingEnv(env, bins=bins, low=low, high=high, n=tilings)

# 샘플 상태 확인
state, _ = tcenv.reset()
print(f"Sample state (Tile Coding): {state}")
# 결과 예: ((idx1_x, idx1_v), (idx2_x, idx2_v), ...) 형태로 4개의 좌표가 나옴

# Q-테이블 초기화 (타일링 개수 차원 추가)
action_values_tc = np.zeros((tilings, 20, 20, 3))


# Tile Coding용 정책 함수
def policy_tc(state, epsilon=0.0):
    if np.random.random() < epsilon:
        return np.random.randint(3)
    else:
        q_vals = np.zeros(3)
        # state는 4개의 (x_idx, v_idx) 튜플을 담고 있음
        for i, (r, c) in enumerate(state):
            # 각 타일링에서 해당 상태의 Q값들을 가져와서 합산 (또는 평균)
            q_vals += action_values_tc[i, r, c]

        return np.random.choice(np.flatnonzero(q_vals == q_vals.max()))


def sarsa_tc(env, action_values, policy, episodes, alpha=0.1, gamma=0.99, epsilon=0.2):
    stats = {"Returns": []}

    # Tile Coding에서는 여러 값을 업데이트하므로 학습률을 타일링 수로 나눠주기도 함
    alpha_tc = alpha / env.n

    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        action = policy(state, epsilon)
        done = False
        total_return = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = policy(next_state, epsilon)

            # 현재 상태의 Q값 (모든 타일링 합)
            current_q = 0
            for i, (r, c) in enumerate(state):
                current_q += action_values[i, r, c, action]

            # 다음 상태의 Q값 (모든 타일링 합)
            next_q = 0
            for i, (r, c) in enumerate(next_state):
                next_q += action_values[i, r, c, next_action]

            # TD Target
            target = reward + gamma * next_q
            td_error = target - current_q

            # 모든 타일링의 Q값 업데이트
            # 각 타일링이 전체 에러의 일부(alpha_tc)만큼 보정
            for i, (r, c) in enumerate(state):
                action_values[i, r, c, action] += alpha_tc * td_error

            state = next_state
            action = next_action
            total_return += reward

        stats["Returns"].append(total_return)

    return stats


# 학습 실행
print("Training with Tile Coding...")
stats_tc = sarsa_tc(
    tcenv, action_values_tc, policy_tc, episodes=20000, alpha=0.1, epsilon=0.1
)
plot_stats(stats_tc)
# 평균 Q값을 이용해 시각화
plot_tabular_cost_to_go(action_values_tc, xlabel="Car Position", ylabel="Velocity")
