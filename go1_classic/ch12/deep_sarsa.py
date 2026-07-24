import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 상태 값들을 텐서로 변환하는 래퍼 클래스...
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        # gymnasium의 Wrapper 클래스는 make로 만든 환경을 받아서 그걸 싸서 생성하는...
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 상태를 FloatTensor로 변환하고 배치 차원 추가(unsqueeze)하고 gpu로...
        return torch.from_numpy(obs).float().unsqueeze(0).to(device), info

    def step(self, action):
        # 텐서로 된 행동을 정수로 변환하여 환경에 전달
        action_item = action.item()
        next_obs, reward, terminated, truncated, info = self.env.step(action_item)

        # 다음 상태 변환 - 실수 텐서로 바꾸고, 배치 차원 추가하고, gpu로
        next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        done = terminated or truncated
        return next_obs, reward, done, truncated, info


# 환경 생성 및 래퍼 적용
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = PreprocessEnv(env)

# 이건 배치 차원 추가 전에 원래 상태와 행동 차원...
state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n
print(f"상태 차원: {state_dims}, 행동 개수: {num_actions}")


# 이게 Q 테이블을 대신할 Q 네트워크...
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 일단 층들은 별거 없이 완전연결 층들로만 구성...
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        # 활성화 함수는 relu...
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# SARSA 맞나? 이름도 Q net이고, 뒤에 경험치도 SARS 까지만...
# 거기다 메인 Q-네트워크와 타겟 Q-네트워크 생성하는 것도 그렇고...
q_network = QNetwork(state_dims, num_actions).to(device)
# 매개변수 그대로 복사하는게 deepcopy인 모양...
target_q_network = copy.deepcopy(q_network).to(device)
# 타겟 네트워크는 학습하지 않는다...eval은 Dropout 끄기, BatchNorm 고정, training 플래그 변경의 역할
target_q_network.eval()
print(q_network)


# ε-greedy 정책
def policy(state, epsilon=0.0):
    # 0~1 uniform dist
    if torch.rand(1).item() < epsilon:
        # 무작위 행동 (탐험)
        return torch.randint(num_actions, (1, 1)).to(device)
    else:
        # Q값이 가장 높은 행동 (활용)
        with torch.no_grad():
            # 상태 s에서 모든 a의 Q값이라는게 결국 그냥 순전파 결과 벡터(이게 Q 함수니까)...
            q_values = q_network(state)
            # 근데 argmax는 gpu로 안보내도 되는 모양...
            # q_values가 QNetwork 모델에서 계산됐고, 이 모델이 gpu에 있으니까 그런 모양...
            return torch.argmax(q_values, dim=-1).view(1, 1)


# 경험 재생 메모리
class ReplayMemory:
    def __init__(self, capacity=100000):
        # double-ended queue 양방향 데이터 처리
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # SARS 까지만 필요한데 이게 왜 SARSA일까...
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 배치 수 만큼 무작위 뽑아서
        batch = random.sample(self.buffer, batch_size)
        # s, a, r, s', done 별로 묶기...
        # s와 s'은 tensor(1,2)가 batch_size개 있는 튜플, a는 tensor(1,1)이 묶여있는 튜플
        # 앞의 1이 배치 차원...
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


memory = ReplayMemory()


# 이게 Deep SARSA 학습...
def deep_sarsa(
    q_network,
    target_q_network,
    policy,
    memory,
    episodes,
    alpha=0.001,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    decay_rate=0.005,
):
    # 옵티마이저를 여기서 지정할 필요가 있나? 아무튼...
    optimizer = optim.AdamW(q_network.parameters(), lr=alpha)
    stats = {"Returns": []}

    # ε 초기화
    epsilon = epsilon_start

    # 에피소드 수만큼 반복해서...
    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        total_return = 0

        # 매 에피소드마다...
        while not done:
            # 1. 행동 선택
            action = policy(state, epsilon)

            # 2. 환경 상호작용
            next_state, reward, done, truncated, _ = env.step(action)

            # 3. 메모리 저장 - 텐서 형상을 맞춰서 저장
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_return += reward

            # 4. 학습 - 배치 크기 이상으로 경험치가 쌓이면...근데 그 뒤에는 매번 학습하나?
            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # 배치 데이터 텐서 변환 -
                # state은 memory에서 action은 policy에서 텐서로 만들어뒀고..
                # tensor(1,2) batch_size개(튜플)를 cat 하면 tensor(batch_size, 2)가 된다...
                batch_states = torch.cat(states)
                batch_actions = torch.cat(actions)
                batch_next_states = torch.cat(next_states)
                # reward는 실수, done은 bool 값 상태...
                # 실수 텐서로 만들고, 두 번째 차원 추가하면 tensor(batch_size, 1)이 된다...
                batch_rewards = (
                    torch.tensor(rewards, device=device).float().unsqueeze(1)
                )
                batch_dones = torch.tensor(dones, device=device).float().unsqueeze(1)

                # 현재 상태의 Q값들 계산하고
                cur_q_vals = q_network(batch_states)
                # 배치 번호는 유지한 채 action이 가리키는 Q 값들만 추린다...그게 각 배치별 현재 Q값
                current_q = cur_q_vals.gather(1, batch_actions)

                # 다음 상태의 행동 a' 선택 (현재 정책 사용)
                with torch.no_grad():
                    # epsilon을 적용하여 다음 행동 선택 (SARSA의 핵심)
                    # 벡터화된 처리를 위해 간단히 argmax 사용 (Deep SARSA 변형)
                    # 또는 DQN처럼 max를 쓸 수도 있지만, 여기서는 SARSA의 철학을 따름
                    # 학습 안정성을 위해 Double DQN 방식이나 Target Net의 max를 쓰기도 함.
                    # 이 튜토리얼에서는 학습 효율을 위해 DQN 스타일의 Target 계산(max)을 차용하되
                    # 엄밀한 SARSA를 원하면 아래 줄을 policy 함수 호출로 바꿔야 함.

                    # 여기서는 Deep SARSA의 일반적 구현인 Target Network + Max Q (DQN) 방식을 사용한다.
                    # 이유: Replay Memory를 쓰면 On-policy인 SARSA의 특성이 희석되기 때문.
                    # ...라고 하는데, 뭔 소린지 헛갈리고...뭔가 부자연스럽다...그냥 DQN을 쓰면 되는거 아닌가?
                    next_q = target_q_network(batch_next_states).max(1)[0].unsqueeze(1)
                    target_q = batch_rewards + gamma * next_q * (1 - batch_dones)

                # 손실 함수는 현재 Q와 목표 Q의 차이...
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                # 이 스텝은 환경의 스텝이 아니고 역전파 결과 적용...
                optimizer.step()

        # 여긴 하나의 에피소드가 끝나면 오는 곳이고...
        # 5. 타겟 네트워크 업데이트 (일정 주기마다 복사하거나 매번 조금씩 업데이트)
        # 여기서는 매 에피소드마다 조금씩 업데이트 (Soft Update)
        tau = 0.05
        for target_param, local_param in zip(
            target_q_network.parameters(), q_network.parameters()
        ):
            # 근데 왜 이건 deepcopy로 똑같이 안만들고 tp + tau(lp - tp) 방식으로 점진적인 업데이트 방법을 쓰는거지?
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

        stats["Returns"].append(total_return)

        # 엡실론 감쇠
        epsilon = max(epsilon_end, epsilon * np.exp(-decay_rate))

    return stats


print("학습 시작...")
# 학습 시간을 고려하여 에피소드 수는 조정 가능 (예: 500)
stats = deep_sarsa(
    q_network, target_q_network, policy, memory, episodes=500, epsilon_start=1.0
)
print("학습 완료!")


# 결과 시각화
def plot_stats(stats):
    plt.figure(figsize=(10, 5))
    plt.plot(stats["Returns"])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


plot_stats(stats)


# 가치함수 시각화
def plot_cost_to_go(env, model, xlabel, ylabel):
    # 그리드 생성
    positions = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], 50
    )
    velocities = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], 50
    )
    X, Y = np.meshgrid(positions, velocities)

    # 각 그리드 포인트에 대한 Q값 예측
    costs = np.zeros_like(X)
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]])
                tensor_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = model(tensor_state)
                # Cost는 -Max Q (보상이 음수이므로)
                costs[i, j] = -torch.max(q_values).item()

    # 시각화
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, costs, cmap="viridis", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Cost-to-Go (-Max Q)")
    ax.set_title("Value Function")
    fig.colorbar(surf)
    plt.show()


plot_cost_to_go(env.unwrapped, q_network, "Position", "Velocity")


# 에이전트 시뮬레이션
def test_agent(env, policy, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0

        img = plt.imshow(env.render())
        plt.axis("off")
        plt.title(f"Test Episode {ep+1}")
        # 연속 그래프 모드 켜기
        plt.ion()

        while not done:
            # 테스트 시에는 탐험 없이 Greedy하게 행동 (epsilon=0)
            action = policy(state, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)

            img.set_data(env.render())
            # 요게 연속으로 그리는데...
            plt.draw()
            plt.pause(0.01)
            step += 1

        # 연속 그래프 모드 끄기
        print(f"Episode {ep+1} finished in {step} steps.")
    plt.ioff()


test_agent(env, policy, episodes=2)
env.close()

# 근데 시뮬레이션 해보니 영 신통칠 않다...tabular 방식으로도 잘 올라갔던 언덕을 못 올라간다...
