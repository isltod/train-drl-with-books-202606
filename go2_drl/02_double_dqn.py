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
from base64 import b64encode
import imageio
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# DQN
class DQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        """
        DQN 네트워크 초기화
        :param obs_size: 입력 상태의 차원
        :param hidden_size: 은닉층 노드 수
        :param n_actions: 출력 행동의 수
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


# 재생버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# 학습 메인 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=128,
        lr=1e-3,
        capacity=100000,
        gamma=0.99,
        batch_size=256,
        sync_rate=10,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_rate = sync_rate

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # 네트워크 초기화
        self.q_net = DQN(obs_size, hidden_size, n_actions).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        # 최적화기 및 손실함수
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # 리플레이 버퍼
        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_t = torch.tensor(np.array([state]), device=device)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    # 하나의 배치(128개)를 샘플링해서 학습
    def train_step(self):
        # 버퍼에 충분한 데이터 없으면 학습 안함
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 상태, 행동, 즉각보상, 종료여부, 다음상태 별로 받아 텐서로...
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)

        # 1. 현재 상태의 Q값 계산 (Main Net)
        state_action_values = self.q_net(states).gather(1, actions)

        # 2. 타겟 Q값 계산 (Double DQN Logic)
        with torch.no_grad():
            # 여기가 기본 DQN과 달라지는 부분인데...행동은 메인 네트워크에서  고르고 가치는 타켓에서 계산한다는 점...
            # (A) 행동 선택: Main Net을 사용하여 다음 상태에서 가장 좋은 행동을 고른다.
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)

            # (B) 다음 상태에서 행동 가치 평가: Target Net을 사용하여 위에서 고른 행동의 가치를 계산한다.
            next_action_values = self.target_q_net(next_states).gather(1, next_actions)

            # 종료된 상태는 (1-done) 마스킹으로 미래 보상이 0
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_action_values
            )

        # 손실 계산 및 역전파
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_training(self, max_episodes=600, max_steps=400):
        total_rewards = []

        for episode in tqdm(range(max_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0

            # 입실론 감쇠
            epsilon = max(0.01, 1.0 - (episode / 200))

            # n-step 학습
            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                # 학습 수행 - 재생버퍼가 충분한지는 train_step에서 확인...
                self.train_step()

                if done:
                    break

            # sync_rate 주기로 타겟 네트워크 동기화
            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

        return total_rewards

    def save_video(self, filename="go2_drl-02_double_dqn"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        # 이렇게 RecordVideo 래퍼 클래스로 감싸주면 알아서 동영상 생성...
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# Double DQN 모델 생성
agent = PytorchWrapper("LunarLander-v3", hidden_size=128, lr=1e-3)

# 학습 시작
print("Double DQN 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Double DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video()
