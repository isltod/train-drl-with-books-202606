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


# Noisy Linear Layer - 여기가 많이 복잡해졌다..
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
        # register_buffer는 역전파 학습되지는 않지만 모델 상태로 같이 저장되는 매개변수...
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
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

        return F.linear(input, weight, bias)
