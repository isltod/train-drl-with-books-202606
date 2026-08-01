import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# Normalized Advantage Function Network
class NafDQN(nn.Module):
    def __init__(self, hidden_size, obs_size, action_dims, max_action):
        """
        NAF 네트워크 초기화
        :param obs_size: 관측 상태 크기
        :param action_dims: 행동 차원 크기
        :param max_action: 행동의 최대 크기 (스케일링용)
        """
        super().__init__()
        # a가 스칼라가 아니라 연속 벡터니까 차원을 정의한다..
        self.action_dims = action_dims
        # 행동을 -1~1 값으로 계산한 후 max_action 값으로 스케일링해서 필요한 값에 맞춘다...
        self.max_action = torch.from_numpy(max_action).to(device)

        # 공통 특징 추출 레이어
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # μ(s): 최적 행동을 예측하는 레이어
        self.linear_mu = nn.Linear(hidden_size, action_dims)

        # V(s): 상태 가치를 예측하는 레이어
        self.linear_value = nn.Linear(hidden_size, 1)

        # L(s) 행렬: Advantage 함수의 곡률(행렬 P)을 결정하는 레이어
        # 하삼각 행렬(Lower Triangular Matrix)의 원소들을 출력함
        # 일단 출력 term이 n(n+1)/2면 n까지 다 더하기인데...
        self.linear_matrix = nn.Linear(
            hidden_size, int(action_dims * (action_dims + 1) / 2)
        )

    # 역전파 안하고 기울기 계산 안하기 데코레이터...
    @torch.no_grad()
    def mu(self, x):
        """최적 행동(Mu) 반환 (테스트 시 사용)"""
        x = self.net(x)
        x = self.linear_mu(x)
        # tanh로 -1~1 사이로 만들고 max_action으로 스케일링
        x = torch.tanh(x) * self.max_action
        return x

    @torch.no_grad()
    def value(self, x):
        """상태 가치 V(s) 반환 (타겟 계산 시 사용)"""
        x = self.net(x)
        x = self.linear_value(x)
        return x

    def forward(self, x, a):
        """Q(s, a) 계산 (학습 시 사용)"""
        # 먼저 특징 추출하고 - x는 (배치 128, 히든 256) 형태가 되고
        x = self.net(x)
        # 행동과 상태 가치를 계산
        # (128, 256) * (256, 2) -> (128, 2) 배치별 행동
        mu = torch.tanh(self.linear_mu(x)) * self.max_action
        # (128, 256) * (256, 1) -> (128, 1) 배치별 상태 가치
        value = self.linear_value(x)

        # L 행렬 구성 - Advantage 함수의 곡률을 결정한다고..
        # 이 층의 이름이 linear_matrix라서 행렬인 줄 착각하기 딱인데, 그게 아니고 wx + b 하는 레이어이고,
        # (128, 256) * (256, 3) -> (128, 3)
        matrix = torch.tanh(self.linear_matrix(x))
        # L 행렬은 (배치 128, 행동 2, 행동 2) shape으로 0으로 초기화하고...
        L = torch.zeros((x.shape[0], self.action_dims, self.action_dims)).to(device)

        # tril_indices - (2, N=3) 형태로 하삼각 행렬 인덱스 가져오기
        tril_indices = torch.tril_indices(
            row=self.action_dims, col=self.action_dims, offset=0
        ).to(device)
        # L 행렬의 배치 부분은 다, 하삼각 부분 row/col에 tanh 적용한 값 넣기...나머지 오른족 위는 다 zeros 상태...
        # L은 (:, [r1, r2, r3], [c1, c2, c3]) matrix는 (:, t1, t2, t3) 형태인데, 각 t들이 좌표 (r, c)에 맞춰 들어가나?
        L[:, tril_indices[0], tril_indices[1]] = matrix

        # 대각 성분은 지수함수를 취해 양수로 만듦 (Positive Definite 보장)
        L.diagonal(dim1=1, dim2=2).exp_()

        # P = L * L^T
        P = L @ L.transpose(2, 1)

        # Advantage 계산: A = -0.5 * (a - mu)^T * P * (a - mu)
        u_mu = (a - mu).unsqueeze(dim=1)  # (Batch, 1, Action)
        u_mu_t = u_mu.transpose(1, 2)  # (Batch, Action, 1)

        adv = -0.5 * u_mu @ P @ u_mu_t
        adv = adv.squeeze(dim=-1)  # (Batch, 1)

        # Q = V + A
        return value + adv


# 재생 버퍼는 간단하게...
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# 학습 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=256,
        lr=1e-3,
        capacity=100000,
        gamma=0.99,
        batch_size=128,
        tau=0.005,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # 타겟 네트워크 소프트 업데이트 비율

        # 환경 생성 (Continuous)
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        max_action = self.env.action_space.high

        # 네트워크 초기화
        self.q_net = NafDQN(hidden_size, obs_size, action_dims, max_action).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # NAF는 주로 MSE 사용

        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state, epsilon):
        """
        Noisy Policy: Mu(s) + Noise
        """
        # state를 (1, obs_size) 형태로 변환 (unsqueeze(0)는 한 번만 수행)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mu = self.q_net.mu(state_t)  # 출력 shape: (1, action_dims)

        # 노이즈 추가
        noise = torch.randn_like(mu) * epsilon
        action = mu + noise

        # 행동 범위 클램핑
        amin = torch.from_numpy(self.env.action_space.low).to(device)
        amax = torch.from_numpy(self.env.action_space.high).to(device)
        action = action.clamp(amin, amax)

        # .squeeze()를 사용해 (1, action_dims) -> (action_dims,)로 변환하여 1차원 배열로 반환
        return action.cpu().numpy().squeeze()

    def soft_update(self, net, target_net):
        """Polyak Averaging을 이용한 타겟 네트워크 업데이트"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )

        # 1. 현재 Q값 계산: Q(s, a)
        # NAF 구조상 a를 입력으로 받아야 함
        current_q = self.q_net(states, actions)

        # 2. 타겟 Q값 계산
        # NAF의 장점: max_a Q(s', a) = V(s')
        # 따라서 별도의 최적화 없이 V(s')만 가져오면 됨
        with torch.no_grad():
            next_v = self.target_q_net.value(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # 3. 손실 계산 및 업데이트
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. 타겟 네트워크 소프트 업데이트
        self.soft_update(self.q_net, self.target_q_net)

        return loss.item()

    def run_training(self, max_episodes=600, max_steps=1000):
        total_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # 탐험 노이즈 크기 조절
            epsilon = max(0.1, 1.0 - (episode / 200))

            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                self.train_step()

                if done:
                    break

            total_rewards.append(episode_reward)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

        return total_rewards

    def save_video(self, filename="naf_dqn_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            # 테스트 시에는 노이즈 없이(epsilon=0) 최적 행동 선택
            action = self.get_action(state, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# 모델 생성
agent = PytorchWrapper("LunarLanderContinuous-v3", hidden_size=256, lr=1e-3)

# 학습 시작
print("NAF (Normalized Advantage Function) 학습을 시작한다...")
history = agent.run_training(max_episodes=200)  # 학습 시간이 오래 걸릴 수 있음
print("학습 완료.")
