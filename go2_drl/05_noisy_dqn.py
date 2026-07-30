import copy
import random
import math
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


# Noisy Linear Layer - 이건 아래 모델에서 사용할 층...
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        """
        Noisy Linear Layer 초기화
        :param in_features: 입력 특징 수
        :param out_features: 출력 특징 수
        :param std_init: 시그마 초기화 상수
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # μ, σ 선언 - empty는 메모리 쓰레기 그대로 두고 크기로만 텐서 만들기,
        # nn.Parameter는 parameters()로 바로 보이고, state_dict()에 저장되도록 매개변수 선언...
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # register_buffer는 역전파 학습되지는 않지만 모델 상태로 같이 저장되는 매개변수...
        # 즉 ε 부분은 학습되지도 않고 reset_noise() 호출할 때마다 다른 값으로 변한다는 말인데...
        # 이 상태에서 위의 μ, σ가 학습되는 걸로 ε-greedy가 학습된다는 말인가?
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # 아래 메서드...
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """파라미터 초기화 (논문에서 제안한 방식)"""
        # 매개변수의 μ는 입력 특성 수의 제곱근으로 나눈 값 범위 내에서 균일 분포 난수로 채우고...
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # σ는 표준편차 초기값을 특성 수의 제곱근으로 나눈 값으로 채우기...
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Factorized Gaussian Noise 생성"""
        # 정규분포에서 size 크기의 난수 만들고
        x = torch.randn(size, device=self.weight_mu.device)
        # sign(x) * |root(x)| 반환
        # sign - 원소 부호에 따라 -1, 0, 1로..., mul - 원소별 곱,
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """노이즈(ε) 다시 샘플링 (에피소드나 스텝마다 호출 가능)"""
        # 위의 FGN 만들고
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # ε_in X ε_out(외적)이 매개변수와 편향의 ε 초기값...
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        """
        Forward Pass
        Training 모드일 때는 노이즈를 섞어서 연산하고,
        Evaluation 모드일 때는 평균값(Mu)만 사용하여 연산한다.
        """
        # 근데 이렇게만 해도 얘가 알아서 μ, σ, ε을 구분해서 학습하나? 맞는 값은 μ라고?
        if self.training:
            # W = mu + sigma * epsilon
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # W = mu (노이즈 제거)
            weight = self.weight_mu
            bias = self.bias_mu

        # 이건 겨우 층이 하나?
        return F.linear(input, weight, bias)


# 이게 Noisy DQN 네트워크...
class NoisyDQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()

        # 일반 선형층 (특징 추출)
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Noisy Linear 층 (행동 결정)
        # 마지막 출력층에 NoisyLinear를 사용하여 행동 가치에 불확실성을 부여한다.
        self.noisy_layer1 = NoisyLinear(hidden_size, hidden_size)
        self.noisy_layer2 = NoisyLinear(hidden_size, n_actions)

    def forward(self, x):
        # 이건 자체적으로 relu 거치고
        x = self.feature_layer(x.float())
        # Noisy 선형 1층은 별도로 relu 거치고...
        x = F.relu(self.noisy_layer1(x))
        # 마지막 Noisy 선형 2층은 활성화 없음...
        x = self.noisy_layer2(x)
        return x

    def reset_noise(self):
        """네트워크 내 모든 Noisy Layer의 노이즈를 재설정한다."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()


# 재생 버퍼 - PER 이전의 단일 형식
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# 학습 클래스...
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=128,
        lr=1e-3,
        capacity=100000,
        gamma=0.99,
        batch_size=64,
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

        # Noisy DQN 네트워크 초기화
        self.q_net = NoisyDQN(obs_size, hidden_size, n_actions).to(device)
        # copy.deepcopy - 모든 중첩 추적해서 원본과 독립적인 객체로 복사 생성
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state):
        """
        NoisyNet에서는 ε-Greedy 필요 없고 네트워크가 내재된 노이즈를 통해 스스로 탐험한다.
        """
        state_t = torch.tensor(np.array([state]), device=device)
        # 이 Q 값에 노이즈가 포함되서 가끔씩 무작위가 더 커진다는 얘기...
        q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        # 데이터 충분할 때만 학습...
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 경험에서 s, a, r, done, s' 별로 샘플링, gpu 텐서로...
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)

        # 학습 단계에서는 노이즈를 매번 다시 샘플링하여 다양성을 준다.
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        # 현재 Q값 계산
        state_action_values = self.q_net(states).gather(1, actions)

        # 타겟 Q값 계산
        with torch.no_grad():
            # argmax 처리를 이런식으로...
            # target_q_net(next_states) - (배치 64, 행동 4)
            # max(1) - 배치별 행동에 따른 max - 0: max, 1: index
            next_action_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            expected_state_action_values = (
                rewards + (1 - dones) * self.gamma * next_action_values
            )

        # 손실 계산 및 업데이트
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_training(self, max_episodes=600, max_steps=400):
        total_rewards = []

        # 에피소드 수만큼 돌면서...
        for episode in tqdm(range(max_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0

            # 최대 스텝 수 내에서 학습...
            for step in range(max_steps):
                # 앞 예제들과 다르게 ε 인수가 없고 내재 매개변수로 처리...
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                self.train_step()

                # 최대 스텝 전에도 에피소드 끝나면 중지하고 다시 시작...
                if done:
                    break

            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        return total_rewards

    def save_video(self, filename="noisy_dqn_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        # 평가 모드에서는 노이즈를 제거하거나 평균값만 사용할 수 있다.
        # PyTorch의 .eval() 모드는 NoisyLinear의 forward에서 처리된다.
        self.q_net.eval()

        while not done:
            with torch.no_grad():
                action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()
        self.q_net.train()  # 다시 학습 모드로 복귀


# Noisy DQN 모델 생성
agent = PytorchWrapper("LunarLander-v3", hidden_size=128, lr=1e-3)

# 학습 시작
print("Noisy DQN 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Noisy DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-05_noisy-dqn")

# 학습 점수는 PER보다 높지 않지만 수렴도 잘 되고 실전에는 잘 맞는듯도 보이고...
