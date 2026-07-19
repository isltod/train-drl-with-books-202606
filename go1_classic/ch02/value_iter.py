from envs import Maze
import matplotlib.pyplot as plt
import numpy as np

# 상태 공간의 크기만큼 가치 테이블 생성
env = Maze()
# DP는 테이블 기반이니까 테이블 만들고 초기값 0으로
# 근데 spaces.Discrete의 요소 수를 반환 하는데...intellij는 이 n이 없다네...
value_table = np.zeros(env.observation_space.n)
print(f"초기 가치 테이블 형태: {value_table.shape}")


# 가치 반복 함수...theta가 수렴 판정값...
def value_iteration(env, gamma=0.99, theta=1e-6):
    # 1. 가치 테이블 초기화
    V = np.zeros(env.observation_space.n)

    # 2. 가치 반복으로 값이 수렴할 때까지 돌기...
    iteration = 0
    while True:
        delta = 0  # 이번 반복에서 가치 변화량의 최댓값

        # 3. 모든 상태 s에 대해 반복하며...
        for s in range(env.observation_space.n):
            # 목표 상태(종료 상태)는 건너뛰니 가치가 0으로 고정 (선택 사항이나 수렴 속도에 도움)
            if s == env.target:
                continue

            # 4. 해당 상태에서 가능한 모든 행동에 대한 기대 가치 계산 (Q-value)
            q_values = []
            for a in range(env.action_space.n):
                # DP는 시뮬레이션 아니고 모든 상태/행동 쌍에 대해서 계산하므로 step 안하고 바로 sars' 확인
                # P[s][a]는 [(prob, next_state, reward, done)] 리스트
                prob, next_s, reward, _ = env.P[s][a][0]

                # 벨만 방정식 적용: R + gamma * V(s')
                # Maze의 전이확률이 결정론적이므로 확률에 대한 Σ 없이 바로 계산
                q_value = prob * (reward + gamma * V[next_s])
                q_values.append(q_value)

            # 5. 한 상태의 행동 다 돌면 최적(max) Q 선택하고
            best_value = np.max(q_values)

            # 변화량 갱신 (수렴 확인용)
            delta = max(delta, np.abs(best_value - V[s]))

            # 6. 가치 테이블 업데이트는 최대 Q로...
            V[s] = best_value

        iteration += 1

        # 변화량이 임계값(theta)보다 작으면 수렴한 것으로 간주하고 종료
        if delta < theta:
            print(f"가치 반복이 {iteration}회 만에 수렴했다.")
            break

    return V


# 알고리즘 실행
optimal_values = value_iteration(env)
print("계산된 최적 가치:\n", optimal_values.reshape(5, 5))


# 정책 추출 함수
def extract_policy(env, value_table, gamma=0.99):
    # 1. 정책 초기화하고
    policy = np.zeros(env.observation_space.n, dtype=int)
    # 2. 모든 상태에 대해 돌면서
    for s in range(env.observation_space.n):
        # 3. Q 값도 초기화하고
        q_values = []
        # 4. 각 행동에 대해 Q-value 재계산
        for a in range(env.action_space.n):
            # 전이확률, 즉각보상, 다음 상태는 환경에서 얻고
            prob, next_s, reward, _ = env.P[s][a][0]
            # 다음 상태를 이용해서 최적 상태가치 구해서 Q값 계산
            q_value = prob * (reward + gamma * value_table[next_s])
            q_values.append(q_value)

        # 5. 가장 높은 가치를 주는 행동 선택 (argmax)
        policy[s] = np.argmax(q_values)

    return policy


optimal_policy = extract_policy(env, optimal_values)
print(
    "추출된 최적 정책 (0:Up, 1:Right, 2:Down, 3:Left):\n", optimal_policy.reshape(5, 5)
)


# 결과 시각화 함수
def plot_results(values, policy):
    # 상태 가치 함수는 히트맵으로...
    plt.figure(figsize=(8, 8))
    plt.imshow(values.reshape(5, 5), cmap="coolwarm", interpolation="none")
    plt.colorbar(label="State Value")
    plt.title("Value Function & Optimal Policy")

    # 정책은 화살표 그리기...dx, dy 순서
    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
    # 모든 상태 돌면서...
    for s in range(25):
        if s == 24:
            continue  # 목표 지점은 화살표 생략
        # 현재 위치
        r, c = divmod(s, 5)
        # 위에서 계산된 해당 위치의 최적 행동
        action = policy[s]
        dx, dy = arrows[action]

        # 화살표 추가 (x는 열, y는 행이므로 순서 주의)
        plt.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc="black", ec="black")

    plt.show()


plot_results(optimal_values, optimal_policy)


# 시뮬레이션...에피소드 실행 함수
def run_episode(env, policy):
    # 환경 등 초기화하고
    state, _ = env.reset()
    done = False
    step = 0
    # trajectory는 s0->a0->r1->s1->...
    path = [state]

    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("Agent Navigation")

    # 연속 그래프 모드 켜기
    plt.ion()

    while (
        not done and step < 20
    ):  # 사실 여기는 최적 정책이므로 최대 스텝 제한이 필요 없는데...
        action = policy[state]
        next_state, reward, done, _, _ = env.step(action)

        state = next_state
        path.append(state)
        step += 1

        img.set_data(env.render())
        # 요게 연속으로 그리는데...
        plt.draw()
        plt.pause(0.2)

    print(f"이동 경로: {path}")
    print(f"총 {step} 스텝 소요")


run_episode(env, optimal_policy)

env.close()
# 연속 그래프 모드 끄기
plt.ioff()

print("시뮬레이션 종료")
