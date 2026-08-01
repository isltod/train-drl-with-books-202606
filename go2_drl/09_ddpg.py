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
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 행위자 네트워크
class Actor(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim, max_action):
        """
        Actor 네트워크: 상태를 입력받아 결정적인(Deterministic) 행동을 출력
        """
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            # 단순하게 2개의 특징추출 완전연결 층 + 완전연결 행동 출력 층...
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),  # 행동 범위를 -1 ~ 1로 제한
        )

    def forward(self, x):
        # 여기서도 행동은 -1~1로 계산한 후 max_action으로 스케일 맞추기...
        return self.net(x) * self.max_action


# 비평가 네트워크
class Critic(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim):
        """
        Critic 네트워크: 상태와 행동을 입력받아 Q-Value를 예측
        """
        super().__init__()
        self.net = nn.Sequential(
            # 이것도 단순하게 2개의 특징추출 층 + 완전연결 Q 계산 층...
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        # 상태와 행동을 결합해서 입력으로 사용...
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# 재생 버퍼도 간단한 버전...
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
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # 타겟 네트워크 소프트 업데이트 비율

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        # 행동 공간 차원 값이 (2,)로 배열로 나오는데 그걸 숫자로 바꿔서 저장
        action_dim = self.env.action_space.shape[0]
        # 앞에서는 v2에서는 action_space.high 사용했는데...
        max_action = self.env.action_space.high[0]  # Action space가 대칭이라고 가정

        # 네트워크 초기화 (Actor, Critic 모드 메인 네트워크와 타겟 네트워크로 나눈다...)
        self.actor = Actor(obs_size, hidden_size, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.critic = Critic(obs_size, hidden_size, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # 최적화기
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(capacity)
        self.max_action = max_action

    def get_action(self, state, noise_scale=0.1):
        """
        행동 선택: Actor의 출력에 노이즈를 더해 탐험 유도
        """
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)

        # self.actor의 순전파는 역전파되는 행동을 반환하고, 여기서는 역전파 없이 행동을 반환한다...
        self.actor.eval()
        with torch.no_grad():
            # 행동이 2차원 벡터로 값이 2개인데 그 중 앞에 값만 취해서 그게 행동이라고? 왜 이렇게 하지?
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()

        # 아무튼 거기에 노이즈 추가 - 정규분포 평균 0, 표준편차 noise_scale * self.max_action
        noise = np.random.normal(0, noise_scale * self.max_action, size=action.shape)
        # 행동은 대칭 가정하에 좌우 max로 자른다...
        action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def soft_update(self, net, target_net):
        """Polyak Averaging: 타겟 네트워크를 천천히 업데이트"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            # τ 비율 만큼만 타겟 네트워크를 업데이트...
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        # 배치 샘플링하고 s, a, r, done, s' 별로 gpu 텐서로 묶고
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
        # 1. Critic 업데이트 - 타겟의 Q와 메인 네트워크의 Q 비교
        # ----------------------------
        with torch.no_grad():
            # 타겟 Actor로 다음 행동 예측
            next_actions = self.actor_target(next_states)
            # 타겟 Critic으로 다음 상태의 Q값 예측
            target_q_values = self.critic_target(next_states, next_actions)
            # 타겟 Q값 계산 (Bellman Equation)
            y_target = rewards + (1 - dones) * self.gamma * target_q_values

        # 현재 Q값 예측
        current_q_values = self.critic(states, actions)

        # Critic 손실 (MSE)
        critic_loss = F.mse_loss(current_q_values, y_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 2. Actor 업데이트 - 메인 네트워크 Q를 최대화하도록 메인네트워크 Q 이동...dQ*da
        # ----------------------------
        # 메인 네트워크 Actor가 예측한 행동에 대한 Q값을 최대화 (경사하강법의 역...)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # 3. 타겟 네트워크 소프트 업데이트
        # ----------------------------
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def run_training(self, max_episodes=600, max_steps=1000):
        total_rewards = []

        # 에피소드 수만큼 돌면서
        for episode in tqdm(range(max_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0

            # 탐험 노이즈 감소 (0.1 -> 0.01)
            noise_scale = max(0.01, 0.1 - (episode / 500))

            # 주어진 스텝 수 내에서 경험...
            for step in range(max_steps):
                action = self.get_action(state, noise_scale)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                # 경험치 충분하면 학습...
                self.train_step()

                if done:
                    break

            total_rewards.append(episode_reward)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Noise: {noise_scale:.3f}"
                )

        return total_rewards

    def save_video(self, filename="ddpg_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            # 테스트 시에는 노이즈 없이(noise_scale=0) 행동 선택
            action = self.get_action(state, noise_scale=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=256,
    actor_lr=1e-4,
    critic_lr=1e-3,
    batch_size=128,
)

# 학습 시작
print("DDPG 학습을 시작한다...")
history = agent.run_training(max_episodes=500)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("DDPG Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-09_ddpg")

# 이건 좀 확실히 안좋아보이는데...
