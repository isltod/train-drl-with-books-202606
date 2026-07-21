from envs import *
from tqdm import tqdm

# 환경 생성
env = Maze()

# Q-테이블 초기화 (5x5x4)
action_values = np.zeros(shape=(5, 5, 4))


# Epsilon-Greedy 정책
def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.randint(4)  # 탐험
    else:
        # greedy 부분인데, ch04와 마찬가지로 (5,5,4)에서 (5,5) 좌표로 선택하면 4개 Q값 배열이 나오는데...
        av = action_values[state[0], state[1]]
        # 그 중에 제일 높은 Q값 방향을 선택...처음에는 다 0으로 시작...
        return np.random.choice(np.flatnonzero(av == av.max()))  # 활용 (동점자 중 랜덤)


def constant_alpha_mc_control(
    policy, action_values, episodes, gamma=0.99, epsilon=0.2, alpha=0.1
):
    """
    alpha: 학습률 (0 < alpha <= 1). 값이 클수록 최근 보상을 더 많이 반영함
    크면 진동하고 작으면 학습이 느려진다...
    """

    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        transitions = []

        # 1. 하나의 에피소드 생성
        while not done:
            action = policy(tuple(state), epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transitions.append((state, action, reward))
            state = next_state

        # 2. 그 하나의 Q-함수 업데이트 (Constant Alpha)
        # 매 에피소드마다 총보상은 0으로 초기화하고 시작...
        G = 0
        # 경험을 역순으로 마지막 출구부터 시작...
        for state_t, action_t, reward_t in reversed(transitions):
            G = reward_t + gamma * G

            # 좌표 unpacking
            r, c = state_t

            # 이 부분만 ch04와 달라지는데...
            # [이전 방식] 1/N 평균 계산 (메모리가 필요함 - sa_returns 리스트에 저장)
            # [현재 방식] 상수 알파를 이용한 증분 업데이트
            # Q_new = Q_old + alpha * (Target - Q_old)
            current_q = action_values[r, c, action_t]
            action_values[r, c, action_t] = current_q + alpha * (G - current_q)

    return action_values


# 기존 Q테이블 초기화
action_values = np.zeros(shape=(5, 5, 4))

# 학습
print(f"학습 시작 (alpha=0.1)...")
constant_alpha_mc_control(policy, action_values, episodes=10000, epsilon=0.2, alpha=0.1)
print("학습 완료!")

# 결과 표시
# Q값 히트맵
plot_action_values(action_values)
# 최적 정책 화살표
plot_policy(action_values)
# 주행 시뮬레이션
test_agent(env, policy, action_values)
