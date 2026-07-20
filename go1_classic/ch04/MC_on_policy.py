from envs import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

env = Maze()

# # 초기 상태 확인
# initial_state, _ = env.reset()
# print(f"관찰 공간(Observation space): {env.observation_space}")
# print(f"행동 공간(Action space) 크기: {env.action_space.n}")
# frame = env.render()
# plt.axis("off")
# plt.imshow(frame)
# plt.show()
#
# 5x5 상태, 4개 행동에 대한 Q값 테이블. 0으로 초기화한다.
action_values = np.zeros(shape=(5, 5, 4))
#
# # 초기 Q 테이블 시각화
# plot_action_values(action_values)


# ε-greedy 정책
def policy(state, epsilon=0.2):
    # 0~1 사이 균일분포에서 난수 선택
    if np.random.random() < epsilon:
        # ε보다 작은 확률로 무작위 행동 선택 (탐험)
        return np.random.randint(4)
    else:
        # 아니면 Q값이 가장 높은 행동 선택하는데...
        # Q 테이블(5,5,4)에서 x,y로 선택하면 4방향 Q값이 나오고
        av = action_values[state[0], state[1]]
        # 거기서 가장 높은 방향 인덱스 선택...
        # 동점일 경우 무작위로 선택하여 편향 방지
        # np.flatnonzero는 1차원 평탄화 후 0이 아닌 값들의 인덱스 반환
        return np.random.choice(np.flatnonzero(av == av.max()))


# 예시: 초기 상태에서 정책 테스트
action = policy((0, 0), epsilon=0.5)
print(f"상태 (0,0)에서 선택된 행동: {action}")


# on-policy 몬테카를로 강화학습 - 정책, Q 테이블, 에피소드 수, γ, ε 받아서...
def on_policy_mc_control(policy, action_values, episodes, gamma=0.99, epsilon=0.2):
    # (상태, 행동) 별로 얻은 총보상(Return)들을 저장할 딕셔너리
    sa_returns = {}

    # 지정한 에피소드만큼 반복해서...
    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        trajectory = []

        # 1. 하나의 에피소드 데이터 생성 (Generate an episode)
        while not done:
            # 현재까지 Q 테이블 이용해서 ε-greedy 정책으로 행동 선택하고
            action = policy(tuple(state), epsilon)
            # 그걸로 step해서 다음 상태, 즉각 보상, 종료 여부 등 받고
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 경험(상태, 행동, 보상) 저장
            trajectory.append((state, action, reward))
            state = next_state

        # 2. Q-함수 업데이트 (Update Q-table)
        G = 0
        # 에피소드의 뒤에서부터 역순으로 총 보상(G_t = R_{t+1} + gamma * G_{t+1}) 계산
        for state_t, action_t, reward_t in reversed(trajectory):
            G = reward_t + gamma * G

            # 상태-행동 쌍을 키로 사용 (리스트나 ndarray는 키가 될 수 없으므로 튜플로 변환)
            sa_pair = (tuple(state_t), action_t)
            # 처음이면 상태-행동 키에 빈 리스트 연결하고
            if sa_pair not in sa_returns:
                sa_returns[sa_pair] = []

            # 반환값 리스트에 추가하고 평균 계산
            sa_returns[sa_pair].append(G)

            # 상태 좌표 unpacking
            r, c = state_t
            # Q(s, a) = Average(Returns(s, a)) 업데이트
            action_values[r, c, action_t] = np.mean(sa_returns[sa_pair])
    # 최종적으로 갱신된 Q 테이블 반환
    return action_values


# 학습 실행
print("학습 시작...")
on_policy_mc_control(policy, action_values, episodes=10000, epsilon=0.2)
print("학습 완료!")

# 결과 확인
plot_action_values(action_values)
plot_policy(action_values)
test_agent(env, policy, action_values)
