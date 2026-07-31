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


# 재생 버퍼 - n-step 적용해서 더 복잡해졌다고...
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
