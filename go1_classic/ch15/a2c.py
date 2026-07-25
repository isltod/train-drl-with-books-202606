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
    # 옵티마이저 설정
    optimizer_actor = optim.AdamW(actor.parameters(), lr=1e-3)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=1e-3)

    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}

    # 초기 상태 (배치 크기: num_envs)
    states, _ = envs.reset()
    states = torch.from_numpy(states).float().to(device)

    # 에피소드 루프 대신 총 업데이트 횟수로 진행하기도 하지만, 여기선 에피소드 개념을 유지하며 진행
    # 단, 병렬 환경 특성상 정확한 '에피소드' 단위 제어가 어려우므로 스텝 단위로 루프를 돔

    # 총 타임스텝 계산 (예: 200 에피소드 분량)
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
            # 행동 선택
            probs = actor(states)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

            # 가치 추정
            value = critic(states)

            # 환경 진행
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
            entropy_term += dist.entropy().mean()

            # 상태 갱신
            states = torch.from_numpy(next_states).float().to(device)

        # 2. 마지막 상태의 가치 추정 (부트스트랩)
        # 마지막 스텝에서 끝났다면(mask=0) 가치는 0, 아니면 critic 예측값
        next_value = critic(states)
        returns = next_value

        actor_loss = 0
        critic_loss = 0

        # 3. 역순으로 어드밴티지 및 손실 계산
        for i in reversed(range(n_steps)):
            # TD Target 계산 (R_t = r_t + gamma * R_{t+1})
            # 마스크를 곱해서 종료된 에피소드의 미래 가치는 0으로 만듦
            returns = rewards_list[i] + gamma * returns * masks[i]

            # 어드밴티지: 타겟 - 가치 (detach를 통해 타겟은 고정)
            advantage = returns - values[i]

            # Actor Loss: -log_prob * advantage (detach)
            # advantage를 detach하지 않으면 critic까지 그라디언트가 흘러가므로 주의
            actor_loss += -(log_probs[i] * advantage.detach()).mean()

            # Critic Loss: MSE(return, value)
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
