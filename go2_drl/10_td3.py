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


# 행위자 네트워크
class Actor(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim, max_action):
        """
        Actor 네트워크: 상태를 입력받아 행동을 출력
        """
        # 모양은 DDPG와 동일...
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * self.max_action


# 비평가 네트워크
class Critic(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim):
        """
        Critic 네트워크: 상태와 행동을 입력받아 Q-Value를 예측
        TD3는 이러한 Critic을 2개 사용한다.
        """
        # 이것도 모양은 대략 DDPG와 같은데, 이걸 2개 만들어서 최소값을 사용한다는 것이 다른 점...
        super().__init__()
        self.net = nn.Sequential(
            # 요기가 다른데 입력으로 상태 + 행동을 받는다?
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# 재생 버퍼
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
        actor_lr=1e-4,
        critic_lr=1e-3,
        capacity=100000,
        gamma=0.99,
        batch_size=128,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_noise = policy_noise  # 타겟 행동에 더할 노이즈 표준편차
        self.noise_clip = noise_clip  # 노이즈 클리핑 범위
        self.policy_delay = policy_delay  # Actor 업데이트 지연 주기
        self.total_it = 0  # 총 업데이트 횟수 카운트

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = self.env.action_space.high[0]

        # 네트워크 초기화 (Actor 1개, Critic 2개)
        self.actor = Actor(obs_size, hidden_size, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.critic1 = Critic(obs_size, hidden_size, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1).to(device)

        self.critic2 = Critic(obs_size, hidden_size, action_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)

        # 최적화기
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr)
        # critic 각각은 3개 층에 w와 b - 6개 텐서가 parameters에 있고, 이걸 리스트로 묶으면 12개의 매개변수 텐서 리스트...
        # 옵티마이저 생성하는데 parameters() 아니라 리스트로 묶은 텐서들을 넣어도 되나?
        # 일단 list(parameters()) 해도 원래 텐서 참조가 전달되는 모양이네...
        self.critic_optimizer = optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

        self.buffer = ReplayBuffer(capacity)
        self.max_action = max_action

    def get_action(self, state, noise_scale=0.1):
        """탐험을 위한 노이즈가 추가된 행동 선택"""
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)

        # self.actor의 순전파 호출하면 역전파되는 행동이 계산되고, get_action으로는 역전파 안되는 행동을 반환한다...
        self.actor.eval()
        with torch.no_grad():
            # actor(state_t)가 배치 차원 있는 (1,2) 형태로 반환하니 그걸 행동 벡터만 있는 (2,) 형태로 바꾸기...
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        # 거기에 노이즈 추가 - 정규분포 평균 0, 표준편차 noise_scale * max_action
        noise = np.random.normal(0, noise_scale * self.max_action, size=action.shape)
        # 대칭 가정하에 좌우 max로 자른다...
        action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        self.total_it += 1
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
            # Target Policy Smoothing: 타겟 행동에 노이즈 추가
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Clipped Double Q-Learning: 두 타겟 Q값 중 작은 값 선택
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y_target = rewards + (1 - dones) * self.gamma * target_q

        # 현재 Q값 예측
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Critic 손실 (두 Critic의 손실 합)
        critic_loss = F.mse_loss(current_q1, y_target) + F.mse_loss(
            current_q2, y_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0

        # ----------------------------
        # 2. Actor 업데이트 (Delayed Update)
        # ----------------------------
        if self.total_it % self.policy_delay == 0:
            # Actor 손실: Q1 값을 최대화
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 타겟 네트워크 소프트 업데이트
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

            actor_loss_val = actor_loss.item()

        return critic_loss.item(), actor_loss_val

    def run_training(self, max_episodes=600, max_steps=1000):
        total_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # 탐험 노이즈
            noise_scale = max(0.1, 0.1 - (episode / 500))

            for step in range(max_steps):
                action = self.get_action(state, noise_scale)
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
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Noise: {noise_scale:.3f}"
                )

        return total_rewards

    def save_video(self, filename="td3_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, noise_scale=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# TD3 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=256,
    actor_lr=3e-4,
    critic_lr=3e-4,
    batch_size=256,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
)
# 학습 시작
print("TD3 (Twin Delayed DDPG) 학습을 시작한다...")
history = agent.run_training(max_episodes=500)
print("학습 완료.")
