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

# GPU 사용 가능 여부 확인 및 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 환경 설정 및 전처리 CartPole-v1 환경을 사용
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 관측값(numpy array)을 텐서로 변환하고 배치 차원(unsqueeze) 추가
        return torch.from_numpy(obs).float().unsqueeze(0).to(device), info

    def step(self, action):
        # 텐서 행동을 정수값으로 변환하여 환경에 전달
        action_item = action.item()
        next_obs, reward, terminated, truncated, info = self.env.step(action_item)

        # 다음 관측값도 텐서로 변환
        next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        done = terminated or truncated
        return next_obs, reward, done, truncated, info


# 환경 생성 및 래퍼 적용
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = PreprocessEnv(env)

state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

print(f"상태 차원: {state_dims}, 행동 개수: {num_actions}")


# Q-네트워크 (Q-Network) 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 은닉층 2개 사용 (128, 64 유닛)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 정책 네트워크(학습 대상)와 타겟 네트워크(정답지) 생성
q_network = QNetwork(state_dims, num_actions).to(device)
target_q_network = copy.deepcopy(q_network).to(device)
target_q_network.eval()  # 타겟 네트워크는 학습 모드 아님 (평가 모드)

print(q_network)


# 경험 리플레이 (Experience Replay)¶
class ReplayMemory:
    def __init__(self, capacity=100000):
        # deque를 사용하여 최대 크기가 넘으면 오래된 데이터부터 삭제
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 버퍼에서 무작위로 배치 크기만큼 샘플링
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


memory = ReplayMemory()


# 책 함수 정의
def policy(state, epsilon=0.0):
    if torch.rand(1).item() < epsilon:
        # 탐험: 무작위 행동 선택
        return torch.randint(num_actions, (1, 1)).to(device)
    else:
        # 활용: Q값이 가장 높은 행동 선택
        with torch.no_grad():
            q_values = q_network(state)
            return torch.argmax(q_values, dim=-1).view(1, 1)


# DQN 알고리즘 구현
def deep_q_learning(
    q_network,
    target_q_network,
    policy,
    memory,
    episodes,
    alpha=0.0001,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    decay_rate=0.005,
):

    optimizer = optim.AdamW(q_network.parameters(), lr=alpha)
    stats = {"Returns": []}
    epsilon = epsilon_start

    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        total_return = 0

        while not done:
            # 1. 행동 선택
            action = policy(state, epsilon)

            # 2. 환경 상호작용
            next_state, reward, done, truncated, _ = env.step(action)

            # 3. 메모리 저장
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_return += reward

            # 4. 학습 (메모리가 배치 크기보다 클 때만 수행)
            if len(memory) > batch_size:
                # 미니배치 샘플링
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # 데이터를 텐서로 변환 및 병합
                batch_states = torch.cat(states)
                batch_actions = torch.cat(actions)
                batch_rewards = (
                    torch.tensor(rewards, device=device).float().unsqueeze(1)
                )
                batch_next_states = torch.cat(next_states)
                batch_dones = torch.tensor(dones, device=device).float().unsqueeze(1)

                # --- DQN 핵심 로직 ---

                # (1) 현재 상태의 Q값 예측: Q(s, a)
                # gather는 action 인덱스에 해당하는 Q값만 추출함
                current_q = q_network(batch_states).gather(1, batch_actions)

                # (2) 다음 상태의 타겟 Q값 계산: max Q(s', a')
                # 타겟 네트워크를 사용하며, 그래디언트가 흐르지 않게 함 (no_grad)
                with torch.no_grad():
                    # dim=1에서 최댓값(max)을 구함. [0]은 값, [1]은 인덱스
                    next_q = target_q_network(batch_next_states).max(1)[0].unsqueeze(1)

                    # 종료 상태(done=1)라면 미래 보상은 0
                    target_q = batch_rewards + gamma * next_q * (1 - batch_dones)

                # (3) 손실 함수 계산 (Smooth L1 Loss가 MSE보다 안정적일 수 있음)
                loss = F.smooth_l1_loss(current_q, target_q)

                # (4) 역전파 및 가중치 업데이트
                optimizer.zero_grad()
                loss.backward()
                # 그래디언트 클리핑 (학습 안정화)
                torch.nn.utils.clip_grad_value_(q_network.parameters(), 100)
                optimizer.step()

        # 5. 타겟 네트워크 소프트 업데이트 (Soft Update)
        # 매 에피소드마다 타겟 네트워크를 조금씩 학습 네트워크 쪽으로 이동
        tau = 0.005
        for target_param, local_param in zip(
            target_q_network.parameters(), q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

        stats["Returns"].append(total_return)

        # 엡실론 감쇠
        epsilon = max(epsilon_end, epsilon * np.exp(-decay_rate))

    return stats


print("DQN 학습 시작...")
# 학습 속도를 위해 500 에피소드 진행
stats = deep_q_learning(q_network, target_q_network, policy, memory, episodes=1000)
print("학습 완료!")


def plot_stats(stats):
    plt.figure(figsize=(10, 5))
    plt.plot(stats["Returns"])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress (CartPole)")
    plt.show()


plot_stats(stats)


def test_agent(env, policy, episodes=2):
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
            # 테스트 시에는 무조건 Greedy 행동 (epsilon=0)
            action = policy(state, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)

            img.set_data(env.render())
            # 요게 연속으로 그리는데...
            plt.draw()
            plt.pause(0.05)
            step += 1

        print(f"Episode {ep+1} finished in {step} steps.")
    plt.ioff()


# 테스트 실행
test_agent(env, policy, episodes=3)
env.close()

# 학습이 되기는 되는거 같다..500번 이상 반복해야 효과가 나온다...
