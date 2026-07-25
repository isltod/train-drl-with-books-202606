import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 병렬 환경 생성 함수
# 이게 아마도 make_env("Acrobot-v1", 1234)와 같이 인수까지 지정된 시그니처를 저장해뒀다가 쓸려고 중첩 함수를 쓰는 모양...
# 일단 이 모양과 아래 SyncVectorEnv() 선언과 같이 받아들여둬야 할 듯...
def make_env(env_name, seed):
    def thunk():
        env = gym.make(env_name)
        # 시드 설정 (필요시)
        # env.reset(seed=seed)
        return env

    return thunk


# CPU 코어 수만큼 환경 생성
num_envs = os.cpu_count() or 1
env_name = "Acrobot-v1"

# Gymnasium의 벡터 환경 사용 (자동으로 병렬 처리)
# SyncVectorEnv: 순차적 실행 (디버깅 용이), AsyncVectorEnv: 병렬 실행 (속도 빠름)
# 내 cpu 코어가 12개니까 여기 envs는 env 12개가 묶인 벡터 구조...
envs = gym.vector.SyncVectorEnv([make_env(env_name, i) for i in range(num_envs)])

# 상태 차원 및 행동 개수 확인 (단일 환경 기준)
# 벡터 환경의 observation_space는 배치 차원이 추가되어 있음
single_observation_space = envs.single_observation_space
single_action_space = envs.single_action_space

print(f"Number of environments: {num_envs}")
# single obs space는 shape으로 받아야하고 single act space는 n으로 받아야 하나?
print(f"State shape: {single_observation_space.shape}")
print(f"Num actions: {single_action_space.n}")


# Actor와 Critic 네트워크 만드는데, 초기 층을 공유한다고도 했는데 일단 분리해서 만든다...
# Actor 네트워크
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 층은 단순히 완전연결 3, relu 활성화, 마지막은 소프트맥스
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


actor = Actor(single_observation_space.shape[0], single_action_space.n).to(device)


# Critic 네트워크
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


critic = Critic(single_observation_space.shape[0]).to(device)


# A2C 알고리즘 - 여기서는 n-step TD 방식으로 학습한다고...
def a2c(envs, actor, critic, episodes, n_steps=5, alpha=1e-4, gamma=0.99):
    # 옵티마이저 설정 - 두 개의 네트워크를 학습시켜야 하니까...
    optimizer_actor = optim.AdamW(actor.parameters(), lr=1e-3)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=1e-3)

    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}

    # 초기 상태(num_envs, single_obs), 근데 텐서가 아니라 넘파이 배열로 넘어오는 모양...
    states, _ = envs.reset()
    states = torch.from_numpy(states).float().to(device)

    # 병렬 구조상 반복은 에피소드 수로 제어하기는 어렵고,
    # n-step TD 이므로 n-step 단위로 이루어지는 업데이트 횟수로 진행하는 것이 자연스럽다...
    # 하지만 여기선 에피소드 개념을 어느정도 유지하며 진행

    # 총 타임스텝 계산 (예: 200 에피소드 분량) - 근데 이게 왜 200 에피소드 분량이지?
    # 예를 들어 에피소드 수가 2면 total_updates = 2*200 // 5 = 80번인데?
    total_updates = episodes * 200 // n_steps

    for update in tqdm(range(total_updates)):
        # 데이터를 저장할 리스트
        log_probs = []
        values = []
        rewards_list = []
        entropy_term = 0
        masks = []

        # 1. n-step 동안 데이터 수집
        for _ in range(n_steps):
            # ch14처럼 범주형 분포로 만들고 거기서 샘플링, 병렬처리니까 tensor(12,3)이 나온다...
            probs = actor(states)
            dist = torch.distributions.Categorical(probs)
            # 요건 하나 선택이니까 tensor(12,)
            actions = dist.sample()

            # 현재 가치 추정, tensor(12,1)
            value = critic(states)

            # 환경 진행 - 모두 ndarray로 (12,6), (12,), (12,)...
            next_states, rewards, terminated, truncated, _ = envs.step(
                actions.cpu().numpy()
            )

            # 종료 여부 마스크 (종료되었으면 0, 아니면 1)
            # terminated나 truncated 중 하나라도 True면 종료로 처리
            dones = np.logical_or(terminated, truncated)
            mask = (
                torch.from_numpy(1 - dones.astype(int)).float().to(device).unsqueeze(1)
            )

            # 저장
            log_probs.append(dist.log_prob(actions))
            values.append(value)
            rewards_list.append(
                torch.from_numpy(rewards).float().to(device).unsqueeze(1)
            )
            masks.append(mask)
            # 엔트로피 값이라는 것도 범주형 분포에서 entropy() 호출하면 그냥 나오네...
            entropy_term += dist.entropy().mean()

            # 다음 상태를 현재 상태로 두고 반복...
            states = torch.from_numpy(next_states).float().to(device)

        # 2. n-step 반복 마지막 상태로 가치 추정 (r + γV(s') 항에 사용할 부트스트랩)
        # 마지막 스텝에서 끝났다면(mask=0) 가치는 0, 아니면 critic 예측값? 마스킹 없는데?
        next_value = critic(states)
        returns = next_value

        actor_loss = 0
        critic_loss = 0

        # 3. n-step 역순으로 각 병렬/스텝별 어드밴티지 및 손실 계산
        for i in reversed(range(n_steps)):
            # TD Target 계산 (R_t = r_t + gamma * R_{t+1})
            # gamma는 스칼라, returns와 mask는 tensor(12,)로 원소별 곱 - 종료된 에피소드의 미래 가치는 0으로 만듦
            returns = rewards_list[i] + gamma * returns * masks[i]

            # 어드밴티지: 타겟 - 가치 (detach를 통해 타겟은 고정)
            advantage = returns - values[i]

            # Actor Loss: -log_prob * advantage (detach) 병렬별로 곱해서 평균...
            # advantage를 detach하지 않으면 value 타고 critic까지 그라디언트가 흘러가므로 주의
            actor_loss += -(log_probs[i] * advantage.detach()).mean()

            # Critic Loss: MSE(return, value) - return은 actor랑 상관없나? detach 안해도 되는 모양...
            critic_loss += F.mse_loss(returns, values[i])

        # 엔트로피 보너스 (선택 사항, 탐험 촉진)
        # actor_loss -= 0.001 * entropy_term

        # 4. 업데이트
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())

        # 시각화를 위한 평균 보상 추적 (정확하지 않음, 근사치)
        if update % 10 == 0:
            stats["Returns"].append(torch.stack(rewards_list).sum().item() / num_envs)

    return stats


# 학습 실행
print("A2C 학습 시작...")
# 학습 스텝 수 조정 (예: 500 단위로 업데이트 반복)
stats = a2c(envs, actor, critic, episodes=1000)
print("학습 완료!")


# 학습 결과 시각화
def plot_stats(stats):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(stats["Actor Loss"])
    axs[0].set_title("Actor Loss")

    axs[1].plot(stats["Critic Loss"])
    axs[1].set_title("Critic Loss")

    axs[2].plot(stats["Returns"])
    axs[2].set_title("Average Returns (approx)")

    plt.show()


plot_stats(stats)


# 학습된 에이전트 시뮬레이션
def test_agent(env_id, actor, episodes=3):
    # 테스트는 단일 환경에서 실행한다...
    env = gym.make(env_id, render_mode="rgb_array")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        total_reward = 0

        img = plt.imshow(env.render())
        plt.axis("off")
        plt.title(f"Test Episode {ep+1}")
        # 연속 그래프 모드 켜기
        plt.ion()

        while not done:
            # 상태 배열을 배치 차원 추가해서 텐서 변환
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # 행동 선택...여기선 greedy
            with torch.no_grad():
                probs = actor(state_tensor)
                action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            img.set_data(env.render())
            # 요게 연속으로 그리는데...
            plt.draw()
            plt.pause(0.05)
            step += 1

        print(f"Episode {ep+1}: Steps={step}, Reward={total_reward}")

    plt.ioff()
    env.close()


test_agent("Acrobot-v1", actor, episodes=3)
# 일단 시간 많이 걸려서 했는데, 성과는 별반 안 좋은 듯...
