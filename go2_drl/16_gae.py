import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """레이어 초기화 함수 (Orthogonal Initialization)"""
    # 직교 초기화...매개변수를 직교 행렬로 채워서 벡터 크기와 방향성을 보존하고 기울기 소실/폭발을 줄인다...
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim):
        super().__init__()
        # 왠지는 모르겠는데, PPO, TRPO는 특징 추출 부분의 활성화 함수를 tanh로 쓰는데...
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.mean_layer = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        # 명시적으로 학습 가능한 파라미터로 설정한다는 건데...그냥 Linear로 해도 학습 되는거 아냐?
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_layer(x)
        # 표준편차는 log 없애고(이럴러면 왜 log라고?),
        # expand_as는 새로 메모리를 만들지 않고 뷰만 늘려서 mean과 같은 크기로 보이게 만들기..
        # 대신 크기가 1인 차원만 늘릴 수 있다고...
        # 근데 이건 expand_as 하나 안하나 mean과 std가 같은 shape인데 왜 하지?
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, x):
        return self.net(x)


# 학습 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.gae_lambda = gae_lambda  # GAE 파라미터
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # 네트워크 생성
        self.actor = Actor(obs_size, hidden_size, action_dim).to(device)
        self.critic = Critic(obs_size, hidden_size).to(device)

        # 최적화기
        self.optimizer = optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5,
        )

    def get_action(self, state):
        """행동 선택 (테스트 용)"""
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        with torch.no_grad():
            # 테스트에서는 샘플링 없이 평균을 행동으로 선택
            mean, _ = self.actor(state_t)
        return mean.cpu().numpy()[0]

    def compute_gae(self, rewards, values, dones, next_value):
        """
        GAE (Generalized Advantage Estimation) 계산
        Advantage = delta + gamma * lambda * next_Advantage
        이게 λ가 0이면 TD고 1이면 MC라는데..왜 그런거냐...암튼 현재는 0.95
        인수는 각 t 별로 이어진 텐서로, r: 즉각 보상들, v: 비평가 평가 상태 가치, d: 종료 여부,
        v'은 n step 다음 t'의 상태 가치 스칼라
        """
        # 일단 일반화된 Advantage 추정값을 0으로 초기화하고...
        advantages = torch.zeros_like(rewards).to(device)
        # λ 추정치 0으로 초기화
        last_gae_lam = 0

        # 즉각 보상의 마지막부터 역순으로...
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 마지막 스텝은 Done 여부를 알 수 없으므로 마스킹은 없고
                next_non_terminal = 1.0 - 0.0
                # 다음 t'의 상태 가치는 마지막 이후 상태 가치로...
                next_val = next_value
            else:
                # 아니면 다음 스텝 종료 조건에 따라 마스킹
                next_non_terminal = 1.0 - dones[t + 1]
                # 다음 t'의 상태 가치는 비평가 평가 상태 가치로...
                next_val = values[t + 1]

            # TD Error: δ = r + γV' - V
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]

            # GAE: A = δ + γλ * A_next? 현재 식은 λ'(A) = δ + γλ 인데?
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam

        # 이렇다는 건, returns는 t 별 Q가 되는데...
        # Returns = Advantage + Value (타겟 가치 함수 학습용)
        returns = advantages + values
        # 반환은 t 별로 GAE 추정과 Q
        return advantages, returns

    def train_step(self):
        """데이터 수집(Rollout) 후 PPO 업데이트"""

        # 1. 데이터 수집 (Rollout)
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        state, _ = self.env.reset()
        done = False

        # n step 단위로 경험을 쌓는다...
        for _ in range(self.n_steps):
            state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )

            # 비평가의 상태 가치는 역전파 없이 계산...
            with torch.no_grad():
                # 행동은 μ와 σ 받아서 정규분포에서 샘플링
                mean, std = self.actor(state_t)
                dist = Normal(mean, std)
                action = dist.sample()
                # 로그 확률도 분포에서 받기 - 근데 행동 별 로그 확률을 더해서 쓰나? (1,2) -> (1,)
                log_prob = dist.log_prob(action).sum(axis=-1)
                value = self.critic(state_t)

            # 행동으로 s', r, 종료여부 받고
            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done_flag = terminated or truncated

            # 그걸 n step 단위로 모은다...
            states.append(state_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done_flag)
            values.append(value)

            state = next_state
            # 혹시 종료됐으면 다시 새로 시작하기로 연결...
            if done_flag:
                state, _ = self.env.reset()

        # 텐서로 이어붙이기 - 2048개 텐서들의 리스트 -> (2048, ...) 텐서로...
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        # 이 두 개는 2048개의 스칼라 리스트라서 그냥 변환
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        values = torch.cat(values).squeeze()

        # 마지막 상태 가치 계산 (GAE용)
        with torch.no_grad():
            # n step의 마지막의 다음 상태 - 끝에 state = next_state 있으니까..
            next_state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )
            # n step 다음 상태의 상태 가치를 비평가가 평가...
            next_value = self.critic(next_state_t).squeeze()

        # 2. GAE 계산 (이 챕터의 핵심)
        # 각 t 별로 이어진 텐서로, r: 즉각 보상들, v: 비평가 평가 상태 가치, d: 종료 여부,
        # v'은 n step 다음 t'의 상태 가치 스칼라
        # 반환 값은 시간 t 별로 묶은 GAE 추정치들과 Q 값들...
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # 배치 처리를 위한 Flatten - 이건 앞에 A2C에서 했던 폼을 유지하기 위해서 있고, 여기선 의미 없지만 이름 맞추기...
        b_states = states
        b_actions = actions
        b_log_probs = log_probs
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # 3. PPO 업데이트 (Epoch 반복)
        indices = np.arange(self.n_steps)

        # n-step 데이터별로 에포크 수 만큼 반복해서 학습...
        for _ in range(self.n_epochs):
            # 시간 상관을 없애려고 뒤섞는 모양...
            np.random.shuffle(indices)

            # 각 n-step/배치, 즉 한 에포크 별로 돌면서
            for start in range(0, self.n_steps, self.batch_size):
                # 이 루프는 batch_size로 건너뛰니까 start는 항상 시작 위치가 되고...끝 위치 정하고...
                end = start + self.batch_size
                idx = indices[start:end]

                # 학습용 미니배치 선택(순서는 위에서 이미 뒤섞었음)
                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_old_log_probs = b_log_probs[idx]
                mb_advantages = b_advantages[idx]
                mb_returns = b_returns[idx]

                # Advantage 정규화 (학습 안정성)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 현재 정책 평가
                # t 상태 넣고, 그 때 a가 아닌 행위자가 예측하는 행동 받아보고
                mean, std = self.actor(mb_states)
                dist = Normal(mean, std)
                # 그 로그 확률과 엔트로피
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)
                # 상태 가치도 그 때 v가 아닌 비평가가 평가하는 상태 가치 받아보고
                new_values = self.critic(mb_states).squeeze()

                # Ratio 계산 (pi_new / pi_old) - 로그 확률이니 빼는 것이 나누기...
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # Surrogate Loss (PPO Clipped Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss - 비평가 평가 상태 가치와 Q 경험의 차를 제곱? 왜 V끼리가 아니라 V와 Q를 비교하지?
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Total Loss - 이런 복잡한 형태로 손실 함수가 정의되네...
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                # 기울기 폭발을 막기 위해서 max_grad_norm을 넘지 않도록 비례해서 줄여주기...
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        # 반환값은 Q 값들의 평균인데...
        return b_returns.mean().item()

    def run_training(self, max_timesteps=200000):
        total_steps = 0
        rewards_history = []

        # 이건 병렬 경험도 아닌데 스텝 수로 관리를 하네? 그냥 A2C와 맞추기 위해서인가?
        while total_steps < max_timesteps:
            # 학습은 메서드 내에서 이뤄지고 반환값은 Q 평균...
            mean_return = self.train_step()
            total_steps += self.n_steps
            rewards_history.append(mean_return)

            if total_steps % (self.n_steps * 5) == 0:
                print(f"Steps: {total_steps}, Mean Return: {mean_return:.2f}")

        return rewards_history

    def save_video(self, filename="gae_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        while not done:
            action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# GAE + PPO 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,  # GAE Lambda 설정
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# 학습 시작
print("GAE (Generalized Advantage Estimation) PPO 학습을 시작한다...")
history = agent.run_training(max_timesteps=300000)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("PPO + GAE Returns")
plt.xlabel("Updates")
plt.ylabel("Mean Return")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-16_gae-ppo")

# 결국 이건 앞의 PPO와 같은 코드였다...책이 정말...
