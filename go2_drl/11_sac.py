import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 행위자 네트워크
class Actor(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim, max_action):
        """
        Actor 네트워크: s 받아서 a의 분포 모수(Mean, Log Std) 출력
        """
        super().__init__()
        self.max_action = float(max_action)

        # 특징 추출 완전연결 2개 층
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 평균과 표준편차 층
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        # 일단 상태 받아서 그대로 평균과 로그 표준편차 계산하고...
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # 표준편차 값의 범위를 제한 (안정적인 학습을 위해)
        log_std = torch.clamp(log_std, -20, 2)
        # 그리고는 다시 로그를 없애? 학습은 로그로 해야하고 중간에 그냥이 필요해서일까? 왜 학습은 로그로 해야하지?
        std = torch.exp(log_std)

        # 가우시안 분포 생성
        dist = Normal(mean, std)

        # Reparameterization Trick (rsample): 샘플링을 하면서도 미분 가능하게 함
        # 계산된 모수(μ, logσ)를 그대로 쓰는게 아니라 그걸 이용해서 정규분포를 만들어 샘플링을 하니,
        # 원래는 역전파를 모수들에 전달할 방법이 없다...
        # rsample은 그걸 ε만 그 분포에 뽑은 후 μ + σ*ε 한 값으로 대체해서 역전파가 연결되게 한다...
        z = dist.rsample()

        # Tanh를 적용하여 행동 범위를 -1 ~ 1로 제한
        action = torch.tanh(z)

        # Log Probability 계산 (Tanh 변환에 따른 보정항 추가) - 이건 왜 하는지 잘 모르겠다...
        # log_prob = log_prob_normal - log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 행위자 순전파는 행동의 평균과 표준편차가 아니라, 행동과 로그확률을 반환...
        return action * self.max_action, log_prob


# 비평가 네트워크
class SoftQNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim):
        """
        Critic 네트워크: 상태 + 행동 -> Q값
        2개의 Q-net을 사용하여 Min 값을 취함
        """
        super().__init__()

        # 근데 같은 걸 2개 만들고 이름만 다르게 주네...
        # Q1 architecture
        self.net1 = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Q2 architecture
        self.net2 = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        # 굳이 s, a를 붙여서 처리할 필요가 있나? 일단 보기는 불편한데 계산 효율은 좋은가보지?
        x = torch.cat([state, action], dim=1)
        # 그냥 같은 입력에 두 네트워크가 각각 계산한걸 반환한다...
        return self.net1(x), self.net2(x)


# 재생 버퍼는 이것도 간단하게...
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
        lr=3e-4,
        capacity=100000,
        gamma=0.99,
        batch_size=256,
        tau=0.005,
        alpha=0.2,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient (Temperature)

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = self.env.action_space.high[0]

        # 네트워크 초기화 - 여기는 Q만 메인과 타겟으로 분리...
        self.actor = Actor(obs_size, hidden_size, action_dim, max_action).to(device)
        self.critic = SoftQNetwork(obs_size, hidden_size, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # 최적화기
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state):
        """행동 샘플링 (학습 중에는 확률적으로, 테스트 시에는 평균값 사용 가능)"""
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        # 학습에는 순전파로 역전파 전달하고, 테스트 등에서는 get_action으로 역전파 없이 행동 샘플링
        with torch.no_grad():
            action, _ = self.actor(state_t)
        # 이것도 배치 차원 없애고 (2,) 형태로 만들기...
        return action.cpu().numpy()[0]

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )

        # ----------------------------
        # 1. Critic 업데이트
        # ----------------------------
        with torch.no_grad():
            # 다음 상태의 행동 샘플링 (Current Policy 사용)
            next_actions, next_log_probs = self.actor(next_states)

            # 타겟 Q값 계산 - 현재 정책으로 다음 Q 계산하고 손실을 이용해 현재 정책을 업데이트...
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Soft Critic Update: 엔트로피 항(alpha * log_prob)을 뺌
            # y = r + gamma * (min_Q - alpha * log_pi)
            soft_target_q = target_q - self.alpha * next_log_probs
            y_target = rewards + (1 - dones) * self.gamma * soft_target_q

        # 현재 Q값
        current_q1, current_q2 = self.critic(states, actions)

        # Critic 손실 - Q1, Q2 손실을 더해서 최소화시킨다...
        critic_loss = F.mse_loss(current_q1, y_target) + F.mse_loss(
            current_q2, y_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 2. Actor 업데이트
        # ----------------------------
        # Reparameterization Trick을 사용하여 행동 샘플링
        # 저장된 경험의 s로 순전파를 해서 경험의 a가 아닌 현재 행위자의 new_a를 받고,
        new_actions, log_probs = self.actor(states)
        # 그걸 비평가가 Q1, Q2로 만든다...
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        # Actor 손실: 엔트로피 최대화 + Q값 최대화 -> 이 Q값을 최대화하도록 행위자를 학습시킨다...
        # Loss = alpha * log_pi - Q
        actor_loss = (self.alpha * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # 3. 타겟 네트워크 업데이트
        # ----------------------------
        self.soft_update(self.critic, self.critic_target)

        # 근데 가만보니 train_step 함수는 손실값들을 반환하는데 이걸 사용하질 않고 있네?
        return critic_loss.item(), actor_loss.item()

    def run_training(self, max_episodes=600, max_steps=1000):
        total_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = self.get_action(state)
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
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        return total_rewards

    def save_video(self, filename="sac_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# SAC 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=256,
    lr=3e-4,
    batch_size=256,
    alpha=0.2,  # Temperature parameter
)

# 학습 시작
print("SAC (Soft Actor-Critic) 학습을 시작한다...")
history = agent.run_training(max_episodes=500)
print("학습 완료.")
