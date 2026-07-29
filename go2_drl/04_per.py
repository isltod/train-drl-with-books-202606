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

        return indices, np.array(weights, dtype=np.float32), samples
