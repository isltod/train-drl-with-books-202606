import copy
import random
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


# 행위자 네트워크
class GradientPolicy(nn.Module):
    def __init__(self, in_features, out_dims, hidden_size=128):
        """
        Actor 네트워크: 상태 -> 행동 분포 (평균, 표준편차)
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc_mu = nn.Linear(hidden_size, out_dims)
        self.fc_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Pendulum-v1의 행동 범위는 -2.0 ~ 2.0이다.
        # tanh의 출력 μ(-1~1)에 2를 곱해 범위를 맞춘다.
        loc = torch.tanh(self.fc_mu(x)) * 2.0

        # 소프트플러스 활성화 함수...1\β * log(1 + exp(β ∙ x)), (기본 β = 1)
        # ReLU의 부드러운 버전으로, 미분하면 시그모이드가 된다고...
        # 표준편차는 항상 양수여야 하므로 softplus를 사용한다.
        scale = F.softplus(self.fc_std(x)) + 1e-3

        return loc, scale


# 비평가 네트워크
class ValueNet(nn.Module):
    def __init__(self, in_features, hidden_size=128):
        """
        Critic 네트워크: 상태 -> 상태 가치 V(s)
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 학습 클래스
class PytorchWrapper:
    # 어떻게 처리되는지는 모르겠는데, 일단 init으로 만드는 self는 하나고, 밑에 학습은 8개가 병렬로 돌아간다...
    def __init__(
        self,
        env_name,
        num_envs=8,
        hidden_size=128,
        policy_lr=1e-4,
        value_lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        n_steps=5,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps  # 몇 스텝마다 업데이트할지 결정

        # 벡터화된 환경 생성 (여러 환경을 병렬로 실행)
        # 근데 A2C는 동기적으로 처리하는거 아닌가? 비동기적 처리면 A3C아닌가?
        self.envs = gym.make_vec(
            env_name, num_envs=num_envs, vectorization_mode="async"
        )

        obs_size = self.envs.single_observation_space.shape[0]
        action_dim = self.envs.single_action_space.shape[0]

        # 네트워크 초기화
        self.policy = GradientPolicy(obs_size, action_dim, hidden_size).to(device)
        self.value_net = ValueNet(obs_size, hidden_size).to(device)

        # 최적화기 설정
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=value_lr)

        # 초기 상태 설정 - 8개 env에서 3차원 상태값을 내놔서 (8,3) 텐서가 된다...
        self.states, _ = self.envs.reset()

    def train_step(self):
        """
        N 스텝 동안 환경과 상호작용하고, 모인 데이터로 학습을 수행한다.
        """
        # 이 리스트들은 하나만 만들어지고...
        state_list, action_list, reward_list, done_list = [], [], [], []
        log_prob_list = []
        entropy_list = []

        # 1. N-step 동안 데이터 수집 (Rollout)
        for _ in range(self.n_steps):
            # 여기는 재생 버퍼를 사용하지 않고, 여러 환경이 실행되니까 state를 인스턴스 변수로 만든다?
            # 맨 처음 여길 들어올 때 envs.reset() 때문에 이미 (병렬 수 8, 상태 3) 텐서로 들어온다...
            # 근데 여러개가 병렬도 도는건 이게 아니고 env 아닌가?
            state_t = torch.tensor(self.states, dtype=torch.float32, device=device)

            # 행위자 네트워크 순전파로 μ, σ 받아서 행동 선택 - (8,3) -> (8,1), (8,1)
            loc, scale = self.policy(state_t)
            dist = Normal(loc, scale)
            # (8,1).sample() -> (8,1)
            action = dist.sample()

            # 로그 확률 및 엔트로피 계산
            # dist.log_prob이 (8,1)로 내놓은 값을 (8,)로 만들어주는 sum...마지막 차원에 값이 1개씩이라 sum의미는 없음...
            # 그럼 그냥 squeeze 같은 걸 안쓰고 왜 헷갈리게 sum이냐...
            log_prob = dist.log_prob(action).sum(dim=-1)
            # 엔트로피는 action에 대해서 구하는게 아니라 분포 자체에 구하는 모양...이것도 sum 의미는 없는...
            entropy = dist.entropy().sum(dim=-1)

            # 환경 상호작용 - step(8,1) -> s'(8,3), r(8,), done(8,), term(8,)
            action_np = action.cpu().numpy()
            next_states, rewards, terminateds, truncateds, _ = self.envs.step(action_np)
            dones = np.logical_or(terminateds, truncateds)

            # 데이터 저장 - 이 반복 끝내면 각 t 별로 한 t에 8개씩 병렬 값들이 들어있는 tensor들이 들어있는 리스트 축적
            state_list.append(state_t)
            action_list.append(action)
            reward_list.append(
                torch.tensor(rewards, dtype=torch.float32, device=device)
            )
            done_list.append(torch.tensor(dones, dtype=torch.float32, device=device))
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)

            self.states = next_states

        # 2. n step에 대한 행동 가치 계산
        with torch.no_grad():
            # 먼저 n step 후의 상태 가치를 계산하는데, 근데 왜 이건 역전파 차단이지?
            next_state_t = torch.tensor(self.states, dtype=torch.float32, device=device)
            # (8,3) -> (8,1) -> (8,)
            next_value = self.value_net(next_state_t).squeeze(-1)

        returns = []
        R = next_value

        # 역순으로 행동 가치 계산 (G_t = r_t + gamma * G_{t+1})
        for t in reversed(range(self.n_steps)):
            # 각 t 마다 8개의 병렬 값이 있으므로 done_list[t]는 (8,)
            R = reward_list[t] + self.gamma * R * (1 - done_list[t])
            returns.insert(0, R)

        # 각 리스트에 5개 텐서, 각 텐서는 (8,)이므로 결과는 (n_steps 5, num_envs 8) 텐서...
        returns = torch.stack(returns)
        # 얘만 (5, 8, 3)
        state_list = torch.stack(state_list)
        log_prob_list = torch.stack(log_prob_list)
        entropy_list = torch.stack(entropy_list)

        # 3. 데이터 평탄화 (Flatten)
        # (n_steps, num_envs, ...) -> (n_steps * num_envs, ...)
        b_states = state_list.view(-1, state_list.shape[-1])
        b_returns = returns.view(-1)
        b_log_probs = log_prob_list.view(-1)
        b_entropy = entropy_list.view(-1)

        # 4. 가치 네트워크(Critic) 학습
        # V(s) 예측
        b_values = self.value_net(b_states).squeeze(-1)

        # Advantage 계산: A(s, a) = Return - V(s)
        advantages = b_returns - b_values

        # Value Loss (MSE) - Advantage가 0이 되도록 학습?
        value_loss = F.mse_loss(b_values, b_returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 5. 정책 네트워크(Actor) 학습
        # Policy Loss = - log_prob * Advantage
        # Advantage는 역전파되지 않도록 detach - 비평가는 행위자와 독립적으로 학습해야 하니까..
        policy_loss = -(b_log_probs * advantages.detach()).mean()

        # 엔트로피 보너스 (탐험 유도)
        entropy_loss = -self.entropy_coef * b_entropy.mean()

        total_policy_loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        return b_returns.mean().item()

    def run_training(self, max_steps=20000, print_interval=1000):
        total_steps = 0
        mean_rewards = []

        # 병렬 학습이라 에피소드 어렵고 스텝 수로 조절...
        while total_steps < max_steps:
            # 학습은 train_step 메서드 안에서 이뤄지고 반환값은 행동 가치 평균값...
            mean_return = self.train_step()
            # 매 반복에서 병렬 수 * n-step 수만큼 도니까...
            total_steps += self.n_steps * self.num_envs

            if total_steps % print_interval < (self.n_steps * self.num_envs):
                print(f"Steps {total_steps}, Average Return: {mean_return:.2f}")
                mean_rewards.append(mean_return)

        self.envs.close()
        return mean_rewards

    def save_video(self, filename="a2c_video"):
        # 비디오 저장을 위한 단일 환경 생성
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        while not done:
            state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                loc, scale = self.policy(state_t)
                # 테스트 시에는 평균값(loc)을 사용하거나 샘플링할 수 있다.
                action = loc.cpu().numpy()[0]

            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# main 블록으로 감싸지 않으면 서브 프로세스 생성할 때 에러 발생...
if __name__ == "__main__":
    # A2C 모델 생성
    agent = PytorchWrapper(
        "Pendulum-v1",
        num_envs=8,  # 8개의 환경을 병렬 실행
        hidden_size=128,
        policy_lr=1e-4,
        value_lr=1e-3,
        n_steps=5,  # 5 스텝마다 업데이트 (TD(5)와 유사 효과)
    )

    # 학습 시작
    # A2C는 병렬 환경 덕분에 스텝 수가 빠르게 증가한다.
    print("A2C (Advantage Actor-Critic) 학습을 시작한다...")
    history = agent.run_training(max_steps=200000)
    print("학습 완료.")

    # 결과 시각화 - 학습 곡선 (Average Return)
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("A2C Average Returns")
    plt.xlabel("Updates")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()

    agent.save_video("go2_drl-14_a2c")

# 이것도 코드가 요란하기만 했지 학습이 잘 안되는거 같은데...
