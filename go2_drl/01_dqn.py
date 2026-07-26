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

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# DQN 생성
class DQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        """
        DQN 네트워크 초기화
        :param obs_size: 입력 상태의 차원 (Observation Space)
        :param hidden_size: 은닉층 노드 수
        :param n_actions: 출력 행동의 수 (Action Space)
        """
        super().__init__()
        # 모델은 단순하게 완전 연결 3층을 relu로 연결...
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        """
        순전파(Forward) 연산
        :param x: 입력 상태 텐서
        :return: 각 행동별 Q-Value
        """
        # 입력 x의 32비트 부동소수점으로 변환...왜지?
        return self.net(x.float())


# 재생 버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        """
        :param capacity: 버퍼의 최대 크기
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        # 현재 크기 반환
        return len(self.buffer)

    def append(self, experience):
        """
        새로운 경험을 버퍼에 추가한다.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        버퍼에서 batch_size만큼 무작위로 경험을 추출한다.
        """
        return random.sample(self.buffer, batch_size)


# 토치 래퍼를 만든다는데 이걸 왜 만드는지...
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
        """
        DQN 에이전트 래퍼 클래스 초기화
        """
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_rate = sync_rate  # 타겟 네트워크 동기화 주기 (에피소드 단위)

        # 환경 생성 (Gymnasium 사용)
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # 네트워크 초기화 (Main Net, Target Net)
        # 입력 노드 수 obs_size, 히든 노드 수 hidden_size, 출력 노드 수 n_actions
        self.q_net = DQN(obs_size, hidden_size, n_actions).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        # 최적화기 및 손실함수
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        # 0 근처에서는 제곱편차 L2처럼, 그 외에는 절대값 편차 L1처럼 움직이는 함수...
        self.loss_fn = nn.SmoothL1Loss()

        # 리플레이 버퍼 100,000
        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state, epsilon):
        """
        입실론-그리디(Epsilon-Greedy) 정책
        """
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            # 상태를 텐서로, argmaxQ(s) 선택...
            state_t = torch.tensor(np.array([state]), device=device)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        """
        하나의 배치를 샘플링하여 학습을 수행한다.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0  # 데이터가 부족하면 학습하지 않음

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)

        # 현재 상태의 Q값 계산
        state_action_values = self.q_net(states).gather(1, actions)

        # 타겟 Q값 계산 (Target Network 사용)
        with torch.no_grad():
            next_action_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            # 종료된 상태면 미래 보상은 0
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
        """
        전체 학습 루프 실행
        """
        total_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # 입실론 감쇠 (탐험 비율 줄이기)
            epsilon = max(
                0.01, 1.0 - (episode / 200)
            )  # 200 에피소드 동안 1.0 -> 0.0에 가깝게

            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 경험 저장
                self.buffer.append((state, action, reward, done, next_state))

                state = next_state
                episode_reward += reward

                # 학습 수행
                self.train_step()

                if done:
                    break

            # 타겟 네트워크 동기화
            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

        return total_rewards

    def save_video(self, filename="rl-video.mp4"):
        """
        학습된 모델로 비디오를 생성한다.
        """
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, video_folder="videos", name_prefix="rl-video"
        )

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, epsilon=0.0)  # 탐험 없이 최적 행동만 선택
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()
        print("비디오 저장 완료")


# 모델 생성
agent = PytorchWrapper("LunarLander-v3", hidden_size=128, lr=1e-3)

# 학습 시작
print("학습을 시작한다...")
history = agent.run_training(max_episodes=500)
print("학습 완료.")
