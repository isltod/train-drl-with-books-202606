import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 텐서 변환 및 배치 차원 추가
        return torch.from_numpy(obs).float().unsqueeze(0).to(device), info

    def step(self, action):
        # 어쨌든 action 텐서에 값으로 행동 선택이 들어있고...
        action_item = action.item()
        next_obs, reward, terminated, truncated, info = self.env.step(action_item)
        # 그 중에 상태는 gpu 텐서로 만들어 반환
        next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        done = terminated or truncated
        return next_obs, reward, done, truncated, info


# 연습 환경은 CartPole, 토치 텐서로 처리하기위해 래퍼로 감싸기...
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = PreprocessEnv(env)

state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

print(f"State dimensions: {state_dims}, Actions: {num_actions}")


# 정책 네트워크 - 여기선 이게 그냥 함수가 아니라 모델이다...
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 여기서도 층들은 완전연결층 3개로 구성..
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        # 마지막으로 소프트맥스 추가...
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


policy_net = PolicyNetwork(state_dims, num_actions).to(device)
print(policy_net)


# 학습되지 않은 상태의 초기 정책 네트워크
def plot_action_probs(probs, labels):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color=["blue", "red"])
    plt.ylabel("Probability")
    plt.title("Action Probabilities")
    plt.ylim(0, 1)
    plt.show()


# 테스트용 상태: 카트가 중간에 정지해 있을 때
neutral_state = torch.zeros(1, 4).to(device)
with torch.no_grad():
    probs = policy_net(neutral_state).cpu().numpy()[0]
# 당연히 왼쪽/오른쪽 행동이 비슷한 확률로 나오고...
# plot_action_probs(probs, labels=["Left", "Right"])


# 이게 REINFORCE 알고리즘
def reinforce(policy_net, episodes, alpha=0.001, gamma=0.99):
    optimizer = optim.AdamW(policy_net.parameters(), lr=alpha)
    stats = {"Returns": []}

    # 주어진 에피소드 수 만큼 돌면서 학습...
    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False

        # 에피소드 동안의 데이터를 저장할 리스트 - 여긴 경험버퍼는 안쓴다...
        log_probs = []
        rewards = []

        # 1. 하나의 에피스도를 돌면서 (Generate an episode)
        while not done:
            # 정책 신경망에서 행동 확률 출력 - 여긴 행동 둘 중에 하나니까 tensor(1,2)
            probs = policy_net(state)

            # 확률 분포에 따라 행동 샘플링
            # Categorical - 확률텐서 또는 비정규화된 로그 확률 텐서 중 하나를 받아서 범주형 확률 분포 개체 생성
            dist = torch.distributions.Categorical(probs)
            # 그 분포에서 확률에 따라 무작위로 하나 뽑기...action은 tensor(1,)
            action = dist.sample()

            # 선택한 행동에 대한 로그 확률 저장 (log_π)...tensor(1,)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # 환경 상호작용 - next_state은 tensor(1,4), 나머지는 실수, 불리언 값
            next_state, reward, done, truncated, _ = env.step(action)

            rewards.append(reward)
            state = next_state

            # (CartPole-v1은 최대 500 스텝이라고...)

        # 이건 Gt가 아니라 그냥 진행과정 보여줄려고 저장하는 모양...
        stats["Returns"].append(sum(rewards))

        # 2. 반환값 G_t 계산 (Calculate Returns)
        # 뒤에서부터 계산하여 할인율 적용: G_t = R_{t+1} + gamma * G_{t+1}
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            # 이러면 다시 앞부터 [G1, G2, ... G_T-1, G_T] 리스트가 될테고...
            returns.insert(0, G)

        # 위가 아니라 여기서 G를 넣으면 이 에피소드의 최대 총보상 G가 되는거 같은데...
        # stats["Returns"].append(G)

        # [G1, G2, ... G_T-1, G_T] 리스트를 텐서로 변환
        returns = torch.tensor(returns, device=device)

        # (선택사항) 반환값 정규화: 학습 안정성을 높임
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 3. 정책 경사 업데이트 (Update Policy)
        policy_loss = []
        # trajectory에서 각 단계별로 log_π_t, G_t 받아서
        for log_prob, G_t in zip(log_probs, returns):
            # 단순히 log_π_t * G_t 이게 loss...
            # 여기서 log_π_t가 방향, G_t가 크기로 Loss를 감소시킨다고...
            policy_loss.append(-log_prob * G_t)

        # 모든 스텝의 손실을 합쳐서 역전파
        optimizer.zero_grad()
        # policy_loss는 tensor(1,) 손실값들이 위 에피소드 스텝 수만큼 들어있는 리스트...
        # 그걸 tensor(#steps, 1)로 쌓아서 합 - 무차원 텐서...torch 스칼라인가?
        # 모든 스텝의 손실을 합해서 그걸로 손실값으로 활용해서 매개변수 개선...
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

    return stats


# 학습
print("REINFORCE 학습 시작...")
# 500 에피소드 정도면 CartPole-v1을 어느 정도 학습함
stats = reinforce(policy_net, episodes=500, alpha=0.001)
print("학습 완료!")


# 학습된 결과 시각화
def plot_stats(stats):
    plt.figure(figsize=(10, 5))
    # 이건 기댓값이 아니라 그냥 반환값들의 합인데...
    plt.plot(stats["Returns"])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Training Progress")
    plt.show()


plot_stats(stats)

# 학습된 정책 확인...
# 상태 정의: [위치, 속도, 각도, 각속도]
left_danger = torch.tensor([[-2.0, 0.0, 0.0, 0.0]]).to(device)  # 왼쪽 끝
right_danger = torch.tensor([[2.0, 0.0, 0.0, 0.0]]).to(device)  # 오른쪽 끝

print("왼쪽 위험 상황에서의 행동 확률:")
with torch.no_grad():
    probs_left = policy_net(left_danger).cpu().numpy()[0]
plot_action_probs(probs_left, ["Left", "Right"])

print("오른쪽 위험 상황에서의 행동 확률:")
with torch.no_grad():
    probs_right = policy_net(right_danger).cpu().numpy()[0]
plot_action_probs(probs_right, ["Left", "Right"])


# 학습된 에이전트 시뮬레이션
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
            # 확률 기반 행동 선택 (혹은 테스트시는 argmax 사용 가능)
            with torch.no_grad():
                probs = policy(state)
                action = torch.argmax(probs, dim=-1)  # Greedy
            state, reward, done, truncated, _ = env.step(action)

            img.set_data(env.render())
            # 요게 연속으로 그리는데...
            plt.draw()
            plt.pause(0.05)
            step += 1

        print(f"Episode {ep+1} finished in {step} steps.")
    plt.ioff()


# 테스트 실행
test_agent(env, policy_net, episodes=3)
env.close()
