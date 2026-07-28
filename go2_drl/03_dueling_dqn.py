import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 여기만 Double DQN과 달라지는 부분이고...
class DuelingDQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        """
        Dueling DQN 네트워크 초기화
        :param obs_size: 입력 상태의 차원
        :param hidden_size: 은닉층 노드 수
        :param n_actions: 출력 행동의 수
        """
        super().__init__()

        # 공통 특징 추출층 (Feature Layer)
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 이점(Advantage) 스트림: 각 행동의 상대적 중요도 학습
        self.advantage_stream = nn.Linear(hidden_size, n_actions)

        # 가치(Value) 스트림: 상태 자체의 절대적 가치 학습
        self.value_stream = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 먼저 공통구조 fc 2개를 relu로 통과시키고
        features = self.feature_layer(x.float())

        # advantage와 상태값으로 분기 - 활성화 없음
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)

        # Dueling Aggregation (결합)
        # 그러니까 V와 (A - A')으로 나눠서 생각할 수 있는데, 뒤를 그냥 A로 두면
        # A가 열라크고 V는 엄청 작을 수도 있고,
        # 반대로 A가 음수로 엄청 작고 V는 엄청 클 수도 있는 문제가 unidentifiable 문제인데...
        # 이렇게 만들면 A - A' 부분이 어느정도 작은 범위 안에 고정되서
        # V를 적당한 범위 내에서 추정할 수 있게 되고, 따라서 A도 대충 맞출 수 있다는 얘기인 듯...
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


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


# 학습 클래스
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

        # Dueling DQN을 Double DQN 구조로 만든다...이 두 기법을 자주 같이 쓴단다...
        self.q_net = DuelingDQN(obs_size, hidden_size, n_actions).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        # 최적화기 및 손실함수
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # 리플레이 버퍼
        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state, epsilon):
        # 늘 그렇듯 ε-greedy 정책...
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_t = torch.tensor(np.array([state]), device=device)
            # 최대화 행동은 메인 네트워크에서 선택...
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        # 데이터 부족하면 학습 안하고 바로 나감
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 아니면 하나의 배치 경험들에 대해서 학습...
        batch = self.buffer.sample(self.batch_size)
        # s, a, r, 종료여부 별로 gpu 텐서로 묶고
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)

        # 현재 상태의 Q값 계산은 메인 네트워크에서...
        state_action_values = self.q_net(states).gather(1, actions)

        # 타겟 Q값도 Double DQN 방식으로 계산
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_action_values = self.target_q_net(next_states).gather(1, next_actions)
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_action_values
            )

        # 손실 계산 및 역전파
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # 여기서는 학습만 하고 q_net/target_q_net 동기화는 run_training에서 진행...
        self.optimizer.step()

        return loss.item()

    def run_training(self, max_episodes=600, max_steps=400):
        total_rewards = []

        for episode in tqdm(range(max_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0

            # ε 감쇄...
            epsilon = max(0.01, 1.0 - (episode / 200))

            for step in range(max_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                # 한 스텝마다 무조건 학습 보내고 버퍼 충분한지는 train_step에서 판단
                self.train_step()

                if done:
                    break

            # Double DQN 방식으로 sync_rate 주기로 target_q_net 동기화
            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

        return total_rewards

    def save_video(self, filename="go2_drl-03_dueling_dqn"):
        # 비디오 촬영은 새 환경 만들어서...
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# Dueling DQN 모델 생성
agent = PytorchWrapper("LunarLander-v3", hidden_size=128, lr=1e-3)

# 학습 시작
print("Dueling DQN 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화
# 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Dueling DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

# 비디오 저장 및 확인
agent.save_video("dueling-dqn")
