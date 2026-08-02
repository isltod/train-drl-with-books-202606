# go1의 REINFORCE를 continuous action space에서 복습
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 정책네트워크 - 평균과 표준편차를 반환하고 그걸로 정규분포에서 샘플링하는 방식...
class GradientPolicy(nn.Module):
    def __init__(self, in_features, out_dims, hidden_size=128):
        """
        정책 네트워크 초기화
        :param in_features: 입력 상태의 차원
        :param out_dims: 출력 행동의 차원
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 평균(Mean)을 예측하는 레이어
        self.fc_mu = nn.Linear(hidden_size, out_dims)
        # 표준편차(Standard Deviation)를 예측하는 레이어
        self.fc_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 평균값 계산
        loc = self.fc_mu(x)
        # 이건 환경에 따라 사용 여부를 결정한다는데 지금 하는 InvertedPendulum 게임은 보통 필요 없다고...
        # loc = torch.tanh(loc)

        # 표준편차 계산 (항상 양수여야 함)
        scale = self.fc_std(x)
        # 소프트플러스 활성화 함수...1\β * log(1 + exp(β ∙ x)), (기본 β = 1)
        # ReLU의 부드러운 버전으로, 미분하면 시그모이드가 된다고...
        scale = F.softplus(scale) + 1e-3  # 0이 되는 것을 방지하기 위해 작은 값 더함

        return loc, scale


# 경험 재생 버퍼 없이 바로 학습 클래스
class PytorchWrapper:
    def __init__(self, env_name, hidden_size=128, lr=1e-3, gamma=0.99):
        self.env_name = env_name
        self.gamma = gamma

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # 정책 네트워크 초기화
        self.policy = GradientPolicy(obs_size, action_dim, hidden_size).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)

    def get_action(self, state):
        """
        상태를 입력받아 확률 분포 생성 후 행동 샘플링
        """
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        # 이게 μ, σ인데 왜 헛갈리게 loc, scale을 이름으로 쓰는 걸까...
        loc, scale = self.policy(state_t)

        # 가우시안 분포 생성
        dist = Normal(loc, scale)

        # 행동 샘플링 - 이 메서드는 학습에는 호출 안하고 비디오 녹화에서만 호출한다...
        action = dist.sample()

        # 로그 확률 계산 (학습 시 필요하므로 저장해둘 수도 있지만, 여기서는 바로 반환하지 않음)
        # REINFORCE는 에피소드 전체 데이터를 모은 후 다시 계산하는 방식이 구현하기 쉬움

        # 배치 차원 없애고 (1,) 형태로 만들기...
        return action.cpu().numpy()[0]

    def calculate_returns(self, rewards):
        """
        감가상각된 누적 보상(Discounted Return) 계산
        """
        returns = []
        R = 0
        # 전형적인 역순 반복으로 최종 보상 계산...
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # 학습 안정화를 위한 정규화 (Mean 0, Std 1)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 여기서 반환하는 총 보상은 각 타임 스텝에서의 Gt를 정규화한 텐서...
        return returns

    def train_episode(self):
        """
        한 에피소드를 실행하고 정책을 업데이트함 (Monte-Carlo Update)
        """
        state, _ = self.env.reset()
        log_probs = []
        rewards = []

        # 1. 하나의 에피소드를 완료하며 데이터 수집
        while True:
            state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )
            # 정책은 평균과 분산으로 받아서 정규분포에서 샘플링...근데 이러면 정책이 역전파로 연결 안되는데?
            loc, scale = self.policy(state_t)
            dist = Normal(loc, scale)

            action = dist.sample()

            # 로그 확률 저장: log pi(a|s)
            log_prob = dist.log_prob(action).sum(dim=-1)
            log_probs.append(log_prob)

            # 환경 상호작용
            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

            if done:
                break

        # 2. 하나의 에피소드 끝낸 후 반환값(Return) 한 번에 계산
        returns = self.calculate_returns(rewards)

        # 3. 정책 손실 계산 (Policy Loss)
        # return이 각 타임스텝 별 Gt가 들어있는 텐서, log_probs는 현재 리스트 안에 개별 log_prob 텐서가 들어있는 상태...
        # Loss = - sum( log_prob * return )
        log_probs = torch.stack(log_probs)
        # 근데 이렇게 Normal에서 log_prob 받아서 그걸 return과 곱하면 역전파가 전달돼서 학습되는 모양이네...
        loss = -(log_probs * returns).sum()

        # 4. 역전파 및 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return sum(rewards)  # 에피소드 총 보상 반환

    def run_training(self, max_episodes=1000, print_interval=200):
        total_rewards = []

        # 에피소드 수만큼 돌면서...
        for episode in range(max_episodes):
            episode_reward = self.train_episode()
            total_rewards.append(episode_reward)

            if episode % print_interval == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        return total_rewards

    def save_video(self, filename="reinforce_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        while not done:
            # 테스트 시에는 평균값(loc)을 행동으로 사용할 수도 있고, 샘플링할 수도 있음
            # 여기서는 샘플링 사용
            action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# 모델 생성
# InvertedPendulum-v4는 Mujoco 환경임. (없다면 Pendulum-v1 등으로 대체 가능)...이라는데,
# 메시지는 deprecated 되었다고 v5를 사용하라는데?
agent = PytorchWrapper("InvertedPendulum-v4", hidden_size=128, lr=1e-3, gamma=0.99)

# 학습 시작
print("REINFORCE (Continuous) 학습을 시작한다...")
history = agent.run_training(max_episodes=1000)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("REINFORCE Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-13_reinforce")

# 뭔가 학습도 안되고...
