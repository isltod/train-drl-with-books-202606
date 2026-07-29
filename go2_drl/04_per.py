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


# PER라지만 Dueling, Double DQN과 함께 사용한다...
class DuelingDQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        """
        Dueling DQN 네트워크 초기화
        :param obs_size: 입력 상태의 차원
        :param hidden_size: 은닉층 노드 수
        :param n_actions: 출력 행동의 수
        """
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.fc_adv = nn.Linear(hidden_size, n_actions)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.feature_layer(x.float())
        adv = self.fc_adv(x)
        value = self.fc_value(x)

        # Dueling Aggregation
        return value + adv - torch.mean(adv, dim=1, keepdim=True)


# 재생버퍼...여기가 달라지는 부분
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        우선순위 리플레이 버퍼 초기화
        :param capacity: 버퍼 크기
        :param alpha: 우선순위 반영 정도 (0: 균등 샘플링, 1: 완전 우선순위 기반)
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0  # 새로운 경험은 최대로 설정하여 한 번은 꼭 학습되게 함

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        # 알고리즘상 초반에는 update 없이 append만 하는데...그 때는 초기값 1일 저장되고,
        # 나중에 update 시작되면 계산된 오차 중 최대값이 저장된다...
        self.priorities.append(self.max_priority)

    def update(self, indices, errors):
        """
        학습 후 TD Error를 기반으로 우선순위를 업데이트한다.
        :param indices: 버퍼 내 인덱스 리스트
        :param errors: TD Error (절대값)
        """
        for idx, error in zip(indices, errors):
            # 일단 오차가 우선순위 값인데...클수록 확률이 높아지는데 맞다...
            priority = error + 1e-5  # 0이 되지 않도록 작은 값 더함
            self.priorities[idx] = priority
            # 현재까지 계산된 오차들 중 제일 큰 값을 저장...
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size, beta=0.4):
        """
        우선순위에 따라 확률적으로 샘플링하고 가중치를 계산한다.
        :param beta: 중요도 가중치 보정 정도 (학습 후반부로 갈수록 1에 가까워져야 함)
        """
        total_items = len(self.buffer)

        # 우선순위를 확률로 변환 (P(i) = p_i^alpha / sum(p_k^alpha))
        prios = np.array(self.priorities, dtype=np.float64)
        # 모든 우선순위값에 α 제곱
        probs = prios**self.alpha
        # 위에서 이미 α 제곱을 다 했으니 그냥 sum해서 나눠주면 확률처럼 작용하는 값이 된다...
        probs /= probs.sum()

        # 확률에 기반하여 인덱스 선택
        indices = np.random.choice(total_items, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 중요도 가중치 계산 (w_i = (N^-1 * P(i)^-1)^beta = (N * P(i))^(-beta))
        weights = (total_items * probs[indices]) ** (-beta)
        weights /= weights.max()  # 안정성을 위해 정규화

        # 가중치는 당연한데, 앞과 다르게 샘플 인덱스도 반환...
        return indices, np.array(weights, dtype=np.float32), samples


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
        alpha=0.6,
        beta_start=0.4,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_rate = sync_rate
        self.beta = beta_start
        self.beta_increment = (
            1.0 - beta_start
        ) / 100000  # Beta를 조금씩 1.0으로 증가시킴

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # 네트워크 초기화 (Dueling DQN을 Double DQN 방식으로)
        self.q_net = DuelingDQN(obs_size, hidden_size, n_actions).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)

        # PER 버퍼 생성
        self.buffer = PrioritizedReplayBuffer(capacity, alpha=alpha)

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_t = torch.tensor(np.array([state]), device=device)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 1. PER 버퍼에서 샘플링 (Beta 적용)
        self.beta = min(1.0, self.beta + self.beta_increment)
        indices, weights, batch = self.buffer.sample(self.batch_size, self.beta)
        # s, a, r, done, s' 별로 텐서로 묶는 건 동일...
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)
        weights = torch.tensor(weights, device=device).unsqueeze(1)  # 가중치 텐서

        # 2. Q값 계산
        state_action_values = self.q_net(states).gather(1, actions)

        # 3. 타겟 Q값 계산 (Double DQN 방식)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_action_values = self.target_q_net(next_states).gather(1, next_actions)
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_action_values
            )

        # 4. TD Error 계산 (우선순위 업데이트용, 기울기 계산 X) - 이게 PER에서 다른 부분...
        td_errors = (
            (state_action_values - expected_state_action_values)
            .abs()
            .detach()
            .cpu()
            .numpy()
        )

        # 5. 버퍼 우선순위 업데이트
        self.buffer.update(indices, td_errors.flatten())

        # 6. 손실 계산 (Importance Sampling Weights 적용)
        loss = (
            weights
            * F.smooth_l1_loss(
                state_action_values, expected_state_action_values, reduction="none"
            )
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
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

                self.train_step()

                if done:
                    break

            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}, Beta: {self.beta:.2f}"
                )

        return total_rewards

    def save_video(self, filename="go2_drl-04_per"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# PER + Dueling DQN 모델 생성
agent = PytorchWrapper(
    "LunarLander-v3", hidden_size=128, lr=1e-3, alpha=0.6, beta_start=0.4
)

# 학습 시작
print("PER (Prioritized Experience Replay) 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("PER Dueling DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("per-dqn")

# 책과 달리 나는 오히려 잘 수렴하며 학습한다...
