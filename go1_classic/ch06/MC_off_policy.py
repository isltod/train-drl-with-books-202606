from envs import *
from tqdm import tqdm


# 행동 정책...
def behavior_policy(state):
    # 4방향 모두 0.25의 확률 (Soft Policy)
    return np.random.randint(4)


# 행동 정책의 확률 b(a|s) = 0.25
prob_behavior = 0.25


# 이게 off-policy MC
def off_policy_mc_control(
    action_values, target_policy, cumulative_weights, episodes, gamma=0.99
):
    # 주어진 에피소드 수만큼 돌면서 MC
    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        transitions = []

        # 1. 행동 정책(b)을 이용하여 하나의 에피소드 생성 (완전 무작위)
        while not done:
            action = behavior_policy(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transitions.append((state, action, reward))
            state = next_state

        # 2. 그 에피소드 경험을 역순으로 가치 및 정책 업데이트
        # 총보상 초기값은 0인데...중요도는 1로 주는건가?
        G = 0.0
        W = 1.0  # 중요도 가중치
        for state_t, action_t, reward_t in reversed(transitions):
            r, c = state_t
            # 총보상은 간단하게 누적곱으로 구하고...
            G = gamma * G + reward_t

            # C(S, A) 업데이트...근데 그냥 계속 더해? 그럼 해당 경우 재방문 횟수로 그냥 1/N 하겠다는 말인데...
            cumulative_weights[r, c, action_t] += W

            # Q(S, A) 업데이트
            q_val = action_values[r, c, action_t]
            action_values[r, c, action_t] += (
                W / cumulative_weights[r, c, action_t]
            ) * (G - q_val)

            # 목표 정책 업데이트 (Greedy)
            best_action = np.argmax(action_values[r, c])
            target_policy[r, c] = best_action

            # 일단 여기부터 이해가 안가는데...
            # 상태 s에서 Q 하나를 업데이트 했고, 거기서 argmax가 이번 행동이 아니면 중요도 이 이후로 이번 샘플은 버린다?
            # 행동 정책이 목표 정책과 다른 행동을 했다면 학습 중단 (가중치 0)
            if action_t != best_action:
                break

            # 이게 importance sample wight π/β 인데...π는 deterministic이므로 해당 행동 확률 1, 아니면 0
            # 중요도 가중치 갱신 (Soft policy 확률인 0.25로 나눔)
            W = W * (1.0 / prob_behavior)

    return action_values, target_policy


env = Maze()
# Q 테이블 초기화 - 초기값을 낮게 설정하여 탐험 유도 가능 (선택사항)
action_values = np.zeros(shape=(5, 5, 4)) - 100.0
# 목표 정책과 누적 가중치인데...
target_policy = np.zeros(shape=(5, 5), dtype=int)
cumulative_weights = np.zeros(shape=(5, 5, 4))

print("학습 시작 (30,000 에피소드)...")
# 오프 폴리시는 수렴에 더 많은 데이터가 필요할 수 있음
off_policy_mc_control(action_values, target_policy, cumulative_weights, episodes=30000)
print("학습 완료!")

# 결과 그래프
plot_action_values(action_values)
plot_policy(target_policy)


# 최적 정책 시뮬레이션
def test_agent(env, policy_table):
    state, _ = env.reset()
    done = False
    step = 0
    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("Test Run")
    # 연속 그래프 모드 켜기
    plt.ion()

    while not done and step < 20:
        # 정책 테이블에서 행동 조회
        r, c = state
        action = policy_table[r, c]
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


test_agent(env, target_policy)
env.close()
