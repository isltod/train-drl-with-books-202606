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


# 모델은 Noisy DQN을 사용하네...
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        """
        Noisy Linear Layer 초기화
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # μ, σ는 학습 매개변수로...
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # ε은 학습 제외
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        # 매개변수, 노이즈 초기화
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        # 매개변수의 μ는 입력 특성 수의 제곱근으로 나눈 값 범위 내에서 균일 분포 난수로 채우고...
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # σ는 표준편차 초기값을 특성 수의 제곱근으로 나눈 값으로 채우기...
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        # sign(x) * |root(x)| 반환
        # sign - 원소 부호에 따라 -1, 0, 1로..., mul - 원소별 곱,
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        # Factorized Gaussian Noise 만들고
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # 그걸 외적해서 ε 초기값으로...
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Training 모드일 때는 노이즈를 섞어서 연산하고,
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Evaluation 모드일 때는 평균값(Mu)만 사용하여 연산한다.
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class DuelingNoisyDQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 이건 앞장의 Noisy DQN과 다른 부분인데...노이즈 층 하나는 Advantage 추정
        self.fc_adv = NoisyLinear(hidden_size, n_actions)
        # 다른 하나는 상태 가치 추정
        self.fc_value = NoisyLinear(hidden_size, 1)

    def forward(self, x):
        x = self.feature_layer(x.float())
        adv = self.fc_adv(x)
        value = self.fc_value(x)
        # 안정성을 위해서 Advantage 평균을 빼주는 방법...
        return value + adv - torch.mean(adv, dim=1, keepdim=True)

    def reset_noise(self):
        self.fc_adv.reset_noise()
        self.fc_value.reset_noise()


# 재생 버퍼 - n-step 적용해서 더 복잡해졌다는데...그냥 같은데? noisy 때문에 복잡해진건데?
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        # PER 구조...
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        # 우선순위 반영 정도 (0: 균등 샘플링, 1: 완전 우선순위 기반)
        self.alpha = alpha
        # 새로운 경험은 최대로 설정하여 한 번은 꼭 학습되게 함
        self.max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def update(self, indices, errors):
        """
        학습 후 TD Error를 기반으로 우선순위를 업데이트한다.
        :param indices: 버퍼 내 인덱스 리스트
        :param errors: TD Error (절대값)
        """
        for idx, error in zip(indices, errors):
            # 일단 오차가 우선순위 값인데...클수록 확률이 높아지는데 맞다...
            priority = error + 1e-5
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
        probs = prios**self.alpha
        probs /= probs.sum()

        # total_items 인덱스들에서 batch_size 만큼 확률에 기반하여 인덱스 선택
        indices = np.random.choice(total_items, batch_size, p=probs)
        # 인덱스로 샘플링하고
        samples = [self.buffer[idx] for idx in indices]

        # 중요도 가중치 계산 (w_i = (N^-1 * P(i)^-1)^beta = (N * P(i))^(-beta))
        weights = (total_items * probs[indices]) ** (-beta)
        weights /= weights.max()

        # 가중치는 당연한데, 앞과 다르게 샘플 인덱스도 반환...
        return indices, np.array(weights, dtype=np.float32), samples


# 학습 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=128,
        lr=1e-4,
        capacity=100000,
        gamma=0.99,
        batch_size=64,
        sync_rate=10,
        n_steps=3,
        alpha=0.6,
        beta_start=0.4,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_rate = sync_rate
        # 이게 달라진 스텝 크기...
        self.n_steps = n_steps

        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / 100000

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # 네트워크 초기화 (Dueling + Noisy)
        self.q_net = DuelingNoisyDQN(obs_size, hidden_size, n_actions).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)

        # PER 버퍼
        self.buffer = PrioritizedReplayBuffer(capacity, alpha=alpha)

    def get_action(self, state):
        """NoisyNet을 사용하므로 Epsilon-Greedy 불필요"""
        state_t = torch.tensor(np.array([state]), device=device)
        # 이 Q 값에 노이즈가 포함되서 가끔씩 무작위가 더 커진다는 얘기...
        q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    # 이게 n-step 보상을 사용해서 재생 버퍼 처리가 복잡해진 부분...
    def play_episode(self):
        """
        한 에피소드를 진행하며 N-step Return을 계산하여 버퍼에 저장한다.
        """
        state, _ = self.env.reset()
        done = False
        transitions = []  # 에피소드 동안의 (s, a, r, d, ns) 저장

        # 1. 에피소드 끝까지 진행하며 데이터 수집
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            # 일단 경험치는 s, a, r, done, s'을 튜플로 묶어서 넣는데...
            transitions.append((state, action, reward, done, next_state))
            state = next_state

        # 2. N-step Return 계산 및 저장
        # R_t^(n) = r_{t+1} + gamma * r_{t+2} + ... + gamma^(n-1) * r_{t+n}
        for i in range(len(transitions)):
            # i번째 스텝부터 i+n번째 스텝까지의 부분 시퀀스 추출 - batch는 n개의 튜플 리스트...
            batch = transitions[i : i + self.n_steps]

            # n-step Discounted Reward 합 계산
            n_step_return = 0
            # k는 순서 인덱스, 튜플은 t에 들어가고
            for k, t in enumerate(batch):
                # 뒤에서부터 돌면서 γ를 누적 곱하는 방식 대신, 앞에서부터 순서 인덱스 0, 1, 2,...를 제곱해줘도 같은 효과...
                # s, a, r...이므로 t[2]가 reward
                n_step_return += (self.gamma**k) * t[2]

            # 현재 상태 (s, a)
            s, a, _, _, _ = transitions[i]

            # N-step 후의 상태 (마지막 상태) 및 종료 여부
            _, _, _, last_done, last_next_state = batch[-1]

            # 버퍼에 저장: (s, a, N-step 보상, 마지막 종료여부, 마지막 s')
            self.buffer.append((s, a, n_step_return, last_done, last_next_state))

        # 이 에피소드에 들어있는 모든 즉각 보상들의 단순 합을 반환값으로...
        return sum([t[2] for t in transitions])

    def train_step(self):
        # 여긴 run_trading에서 이미 이 조건을 확인하니까 필요 없는데?
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 중요도 가중치 보정 정도를 1로 조금씩 늘려가며 샘플링
        self.beta = min(1.0, self.beta + self.beta_increment)
        # 경험치는 (s, a, N-step 보상, 마지막 종료여부, 마지막 s') 튜플로 구성...
        indices, weights, batch = self.buffer.sample(self.batch_size, self.beta)
        # s, a, r, done, s' 별로 텐서로 묶는 건 동일...
        states, actions, returns, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        returns = torch.tensor(returns, device=device).unsqueeze(1)  # N-step Return
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), device=device)
        weights = torch.tensor(weights, device=device).unsqueeze(1)

        # 노이즈 리셋
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        # 현재 Q값 순전파 계산
        state_action_values = self.q_net(states).gather(1, actions)

        # 타겟 Q값 계산 (N-step TD Target)
        with torch.no_grad():
            # max Q_target 순전파 계산
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_action_values = self.target_q_net(next_states).gather(1, next_actions)
            # Target = R^(n) + gamma^n * max Q_target(s_{t+n}, a'), 종료 상태면 0
            # 여기 return은 경험치 저장할 때 γ를 누적곱해서 만든 n-step return,
            # 그 뒤 next_action_value는 그 n-step 이후의 행동 가치 - γ^n을 곱한다...
            expected_state_action_values = (
                returns + (1 - dones) * (self.gamma**self.n_steps) * next_action_values
            )

        # TD Error 계산 (우선순위 업데이트용, 기울기 계산 X) - 이게 PER에서 다른 부분...
        td_errors = (
            (state_action_values - expected_state_action_values)
            .abs()
            .detach()
            .cpu()
            .numpy()
        )
        # 버퍼 우선순위 업데이트
        self.buffer.update(indices, td_errors.flatten())

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

    def run_training(self, max_episodes=600):
        total_rewards = []

        # 에피소드 수만큼 돌면서...
        for episode in tqdm(range(max_episodes)):
            # 위에서 정의한 메서드로, 한 에피소드 다 돌고,
            # 경험치를 (s, a, N-step 보상, 마지막 종료여부, 마지막 s') 튜플로 정리해서 버퍼에 저장
            episode_reward = self.play_episode()

            # 학습 수행 (데이터가 어느 정도 쌓인 후)
            if len(self.buffer) > self.batch_size:
                # 이건 왜 에피소드당 여러 번 학습이 가능한거지? 배치가 작아서 그런건가?
                for _ in range(10):
                    self.train_step()

            # 타겟 네트워크 동기화
            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Beta: {self.beta:.2f}"
                )

        return total_rewards

    def save_video(self, filename="n_step_dqn_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        self.q_net.eval()  # 평가 모드 (노이즈 제거)

        while not done:
            with torch.no_grad():
                action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()
        self.q_net.train()


# N-step DQN 모델 생성 (N=3)
agent = PytorchWrapper("LunarLander-v3", hidden_size=128, lr=1e-4, n_steps=3)

# 학습 시작
print(f"N-step DQN (N={agent.n_steps}) 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title(f"{agent.n_steps}-step DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-06_n_step_dqn")

# 이건 성과가 앞에 것들보다 별로 좋지 않아보이는데...
