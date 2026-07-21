from envs import *
from tqdm import tqdm


# ---------------------------------------------------------
# 2. SARSA 알고리즘
# ---------------------------------------------------------
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q-테이블 초기화
    q_table = np.zeros(shape=(5, 5, 4))

    # 주어진 에피소드 수만큼 반복해서 학습...
    for episode in range(episodes):
        state, _ = env.reset()

        # 1. 초기 행동 선택 (Epsilon-Greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        done = False

        # 각 에피소드 별로 반복해서 Q값 업데이트...
        while not done:
            # 2. 행동 실행 및 관측
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 다음 행동 선택 (On-policy: 현재 정책으로 다음 행동 A'를 미리 선택)
            if np.random.random() < epsilon:
                next_action = np.random.randint(4)
            else:
                next_action = np.argmax(q_table[next_state[0], next_state[1]])

            # 현재 Q와 다음 Q
            curr_q = q_table[state[0], state[1], action]
            next_q = q_table[next_state[0], next_state[1], next_action]

            # 목표 지점에 도달했다면 다음 가치는 0, 아니면 target 값 계산해서...
            if done:
                target = reward
            else:
                target = reward + gamma * next_q

            # 4. SARSA 업데이트 식 적용
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            q_table[state[0], state[1], action] = curr_q + alpha * (target - curr_q)

            # 5. 상태 및 행동 변수 갱신
            state = next_state
            action = next_action

    return q_table


# 환경 생성
env = Maze()
print("학습 시작 (SARSA)...")
q_table = sarsa(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1)
print("학습 완료!")

# 결과 시각화
plot_action_values(q_table)
plot_policy(q_table)


# 결과 정책으로 에이전트 테스트
def test_agent(env, q_table):
    state, _ = env.reset()
    done = False
    step = 0
    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("SARSA Agent Test Run")
    # 연속 그래프 모드 켜기
    plt.ion()

    while not done and step < 20:
        # Greedy Action
        r, c = state
        action = np.argmax(q_table[r, c])

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        img.set_data(env.render())
        # 요게 연속으로 그리는데...
        plt.draw()
        plt.pause(0.2)
        step += 1
    print("테스트 종료")


test_agent(env, q_table)
