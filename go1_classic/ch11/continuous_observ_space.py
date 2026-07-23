import numpy as np
import matplotlib.pyplot as plt


# 시드 고정 함수
def seed_everything(env, seed=42):
    np.random.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


# 가치 함수(Cost-to-Go) 시각화 함수
def plot_tabular_cost_to_go(action_values, xlabel, ylabel):
    # 각 상태의 최대 가치(Max Q) 계산
    if len(action_values.shape) == 4:
        # Tile Coding의 경우 (Tilings, Dim1, Dim2, Action)
        values = np.max(action_values.mean(axis=0), axis=2)
    else:
        # State Aggregation의 경우 (Dim1, Dim2, Action)
        values = np.max(action_values, axis=2)

    # -Max Q를 Cost로 변환 (Mountain Car는 보상이 음수이므로, -Max Q는 Cost-to-Go와 유사)
    cost_values = -values

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    x = np.arange(cost_values.shape[0])
    y = np.arange(cost_values.shape[1])
    X, Y = np.meshgrid(y, x)  # X, Y 축 매핑 주의

    surf = ax.plot_surface(X, Y, cost_values, cmap="viridis", edgecolor="none")
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.set_zlabel("Cost (-Max Q)")
    ax.set_title("Cost to Go Function")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# 학습 통계 시각화 함수
def plot_stats(stats):
    plt.figure(figsize=(10, 5))
    plt.plot(stats["Returns"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns over Time")
    plt.show()


# 에이전트 테스트 함수
def test_agent(env, policy, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        img = plt.imshow(env.render())
        plt.axis("off")
        plt.title(f"Test Episode {ep+1}")

        while not done:
            action = policy(state, epsilon=0.0)  # Greedy
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            img.set_data(env.render())
            plt.pause(0.01)  # 애니메이션 효과
            step += 1
        print(f"Episode {ep+1} finished in {step} steps.")
    plt.close()
