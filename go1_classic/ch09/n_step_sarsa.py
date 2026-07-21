from envs import *
from tqdm import tqdm


# ---------------------------------------------------------
# 2. n-step SARSA 알고리즘
# ---------------------------------------------------------
def n_step_sarsa(env, episodes, n=3, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Q-테이블 초기화
    q_table = np.zeros(shape=(5, 5, 4))

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()

        # 초기 행동 선택
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        # n-step을 저장할 버퍼 (states, actions, rewards)
        states = [state]
        actions = [action]
        rewards = [0.0]  # R_0는 없으므로 0으로 채움 (인덱스 맞춤용)
        # 종료 시점 초기화...양의 무한대, 음의 무한대는 float("-inf")
        # 일단 무한대로 해놓고, done 발생하면 거기에 맞춰 줄여서 루프 나오게 처리...
        T = float("inf")
        t = 0  # 현재 타임스텝

        while True:
            # 1. 하나의 에피소드를 반복하면서 행동 수행 및 저장
            if t < T:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(next_state)
                rewards.append(reward)

                if done:
                    # 이 에피소드의 종료 상태 나오면 다음 상태/행동 처리 없이 Q값 갱신으로...
                    # 예를 들어 t=9에 여길 오면 T=10
                    T = t + 1
                else:
                    # 아니고 에피소드 중이면 계속해서 다음 행동 선택 (On-policy)
                    if np.random.random() < epsilon:
                        next_action = np.random.randint(4)
                    else:
                        next_action = np.argmax(q_table[next_state[0], next_state[1]])
                    actions.append(next_action)
                    action = next_action  # 다음 루프를 위해 갱신

            # 2. 업데이트해야 할 시점(tau) 계산
            # t=0에서 시작, n=3이므로, tau = -2,-1,0 순서로 변하고 아래에서 tau>=0 조건을 테스트하니 3번 반복마다...
            # 예를 들어 t=9면 tau=7
            tau = t - n + 1

            # 일단 3번 이상, 최소 경험 데이터가 채워졌다면 그 이후로는 매 step마다 Q 값 갱신하는데...
            if tau >= 0:
                # 3. 반환값 G 계산
                G = 0
                # tau + 1 = 1 부터 종료 전이면 T=∞이므로 1~n개까지,
                # 종료면 T=t+1이므로 n보다 적은 t+1까지만 사용(경계처리)해서 총보상 계산
                # 예를 들어 t=9면 tau+1=8, tau+n=10, T=10, 다음에는 t=10면 tau+1=9, tau+n=11, T=10이 돼서
                # 어쨌거나 종료까지만 보게 된다...
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]

                # 4. 부트스트랩 (종료 상태가 아니라면 n번째 후의 Q값 추가)
                # 종료 상태가 한 번 뜨면 n step 학습 기간에는 T가 고정되고 tau+n보다 항상 작거나 같다...
                if tau + n < T:
                    s_n = states[tau + n]
                    a_n = actions[tau + n]
                    G += (gamma**n) * q_table[s_n[0], s_n[1], a_n]

                # 현재 시점의 Q 값 찾고
                s_tau = states[tau]
                a_tau = actions[tau]
                curr_q = q_table[s_tau[0], s_tau[1], a_tau]

                # 5. Q값 업데이트
                q_table[s_tau[0], s_tau[1], a_tau] = curr_q + alpha * (G - curr_q)

            # 전체 루프 종료 조건: 업데이트 시점 tau가 종료 시점 T-1에 도달했을 때
            # 예를 들어 tau==T-1 상태면 더 이상 남은 학습 데이터가 없다...
            if tau == T - 1:
                break

            t += 1

    return q_table


# 환경 생성
env = Maze()

print("학습 시작 (n=3 SARSA)...")
# n=3으로 설정
q_table = n_step_sarsa(env, episodes=10000, n=3, alpha=0.1, gamma=0.99, epsilon=0.1)
print("학습 완료!")

# 결과 시각화
plot_action_values(q_table)
plot_policy(q_table)


# 최적 정책 에이전트 시뮬레이션
def test_agent(env, q_table):
    state, _ = env.reset()
    done = False
    step = 0
    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("n-step SARSA Agent Test Run")
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
    # 연속 그래프 모드 끄기
    plt.ioff()
    print("테스트 종료")


test_agent(env, q_table)
env.close()
