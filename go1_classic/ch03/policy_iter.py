from envs import Maze
import matplotlib.pyplot as plt
import numpy as np


def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    """
    주어진 정책(policy)에 대한 가치 함수(V)를 계산한다.
    """
    # 1. 상태 가치를 0으로 초기화
    V = np.zeros(env.observation_space.n)

    # 2. 상태 가치 함수를 반복 갱신하면서 임계값보다 작으면 종료
    while True:
        delta = 0
        # 3. 모든 상태에 대해 반복해서...
        for s in range(env.observation_space.n):
            # 4. 현재 정책이 선택하는 행동 가져오기
            action = policy[s]

            # 5. 알고있는 전이확률 P에서 전이 정보 가져오기 (결정론적 환경이므로 항목은 1개)
            prob, next_state, reward, done = env.P[s][action][0]

            # 6. 벨만 기대 방정식 적용
            # 마지막의 (1 - done) 항으로, 목표 상태인 경우 다음 상태 가치는 reward만 남음
            new_v = prob * (reward + gamma * V[next_state] * (1 - done))

            # 7. 변화량 갱신 - delta를 계속 줄여나간다...
            delta = max(delta, np.abs(new_v - V[s]))
            V[s] = new_v

        # 가치 함수의 변화가 임계값보다 작으면 수렴한 것으로 판단
        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    """
    주어진 가치 함수(V)를 바탕으로 더 나은 정책(policy)을 생성한다.
    """
    # 1. 모든 상태에 대해서 정책 초기화...
    policy = np.zeros(env.observation_space.n, dtype=int)

    # 2. 모든 상태에 대해서 반복하면서...
    for s in range(env.observation_space.n):
        # 3. 행동 가치를 초기화하고
        q_values = []

        # 4. 가능한 모든 행동에 대해 반복하면서...
        for a in range(env.action_space.n):
            # 5. 알고있는 P를 통해서 전이 정보 받고...
            prob, next_state, reward, done = env.P[s][a][0]
            # 6. Bellman expectation 식을 이용해서 행동 가치 계산...
            q_val = prob * (reward + gamma * V[next_state] * (1 - done))
            q_values.append(q_val)

        # 7. 가장 높은 Q-value를 가진 행동을 선택 (Argmax)
        policy[s] = np.argmax(q_values)

    return policy


# 정책 평가와 개선을 묶는 정책 반복 함수...
def policy_iteration(env, gamma=0.99):
    # 1. 임의의 정책 초기화 (모두 0번 행동 등으로 초기화하거나 랜덤)
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)

    step = 0
    while True:
        # 2. 정책 평가: 현재 정책에 대한 가치 함수 계산
        V = policy_evaluation(env, policy, gamma)

        # 3. 정책 발전: 가치 함수를 기반으로 새로운 정책 생성
        new_policy = policy_improvement(env, V, gamma)

        # 4. 정책이 변했는지 확인 - 이산 정책이므로 같은지를 확인할 수 있다..
        if np.array_equal(new_policy, policy):
            print(f"정책 반복이 {step}회 만에 수렴했다.")
            break

        policy = new_policy
        step += 1

    return policy, V


# 알고리즘 실행
env = Maze()
optimal_policy, optimal_values = policy_iteration(env)

print("\n[최적 정책 (0:Up, 1:Right, 2:Down, 3:Left)]")
print(optimal_policy.reshape(5, 5))

print("\n[최적 가치 함수]")
print(optimal_values.reshape(5, 5))


# 상태와 행동 가치 함수 히트맵...이것도 ch02와 같은 코드...
def plot_policy(values, policy):
    plt.figure(figsize=(8, 8))
    # 가치 함수 히트맵
    plt.imshow(values.reshape(5, 5), cmap="coolwarm", interpolation="none")
    plt.colorbar(label="Value")
    plt.title("Optimal Value & Policy")

    # 정책 화살표
    arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}  # dx, dy
    for s in range(25):
        if s == 24:
            continue  # 목표 위치
        r, c = divmod(s, 5)
        action = policy[s]
        dx, dy = arrows[action]
        plt.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc="black", ec="black")

    plt.show()


plot_policy(optimal_values, optimal_policy)


# 에이전트 주행 시뮬레이션...이것도 ch02와 같은 함수인데 이게 더 간단하네...
def run_simulation(env, policy):
    state, _ = env.reset()
    done = False
    step = 0

    img = plt.imshow(env.render())
    plt.axis("off")
    plt.title("Agent Simulation")
    plt.ion()

    while not done and step < 20:
        action = policy[state]
        state, reward, done, _, _ = env.step(action)

        img.set_data(env.render())
        plt.draw()
        plt.pause(0.2)
        step += 1

    plt.ioff()
    print("시뮬레이션 종료")


run_simulation(env, optimal_policy)
env.close()
