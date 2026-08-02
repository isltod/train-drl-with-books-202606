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
        # 대신 크기가 1인 차원만 늘릴 수 있다고...(1, 행동) -> (히든, 행동)으로...
        # 배치 크기에 맞게 확장? 이라고? 나중에 다시 보자...
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


class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,  # GAE 파라미터
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
        self.gae_lambda = gae_lambda
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

        # 최적화기 (Actor와 Critic 파라미터를 함께 최적화)
        self.optimizer = optim.AdamW(
            # 이렇게 리스트로 더해서 넣으면 다 관리하는 모양...
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
        """
        # 일단 일반화된 Advantage 추정값을 0으로 초기화하고...
        advantages = torch.zeros_like(rewards).to(device)
        # 마지막 λ는 0?
        last_gae_lam = 0

        # 즉각 보상의 마지막부터 역순으로...
        for t in reversed(range(len(rewards))):
            # 마지막 타임 스텝이면...
            if t == len(rewards) - 1:
                # 마지막 스텝은 Done 여부를 알 수 없으므로 마스킹은 없고, GAE 추정은 그냥 다음 상태 가치?...
                next_non_terminal = 1.0 - 0.0
                next_val = next_value
            else:
                # 아니면 다음 스텝 종료 조건에 따라 마스킹---------------------여기 보던 중...
                next_non_terminal = 1.0 - dones[t + 1]
                # 근데 next_value와 values는 어떻게 다른거냐?
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    def train_step(self):
        """데이터 수집(Rollout) 후 PPO 업데이트"""

        # 1. 데이터 수집 (Rollout)
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        state, _ = self.env.reset()
        done = False

        for _ in range(self.n_steps):
            state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )

            with torch.no_grad():
                mean, std = self.actor(state_t)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)
                value = self.critic(state_t)

            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done_flag = terminated or truncated

            states.append(state_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done_flag)
            values.append(value)

            state = next_state
            if done_flag:
                state, _ = self.env.reset()

        # 텐서 변환
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        values = torch.cat(values).squeeze()

        # 마지막 상태 가치 계산 (GAE용)
        with torch.no_grad():
            next_state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )
            next_value = self.critic(next_state_t).squeeze()

        # 2. GAE 계산
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Flatten (배치 처리를 위해)
        b_states = states
        b_actions = actions
        b_log_probs = log_probs
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # 3. PPO 업데이트 (Epoch 반복)
        indices = np.arange(self.n_steps)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

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
                mean, std = self.actor(mb_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)
                new_values = self.critic(mb_states).squeeze()

                # Ratio 계산 (pi_new / pi_old)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # Surrogate Loss (Clipped Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Total Loss
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        return b_returns.mean().item()

    def run_training(self, max_timesteps=200000):
        total_steps = 0
        rewards_history = []

        # 이건 병렬 경험도 아닌데 스텝 수로 관리를 하네? 에피소드 done이면 어떻게 처리가 되지? train_step() 보고와야...------------
        while total_steps < max_timesteps:
            mean_return = self.train_step()
            total_steps += self.n_steps
            rewards_history.append(mean_return)

            if total_steps % (self.n_steps * 5) == 0:
                print(f"Steps: {total_steps}, Mean Return: {mean_return:.2f}")

        return rewards_history

    def save_video(self, filename="ppo_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        while not done:
            action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# PPO 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=128,
    lr=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# 학습 시작
print("PPO (Proximal Policy Optimization) 학습을 시작한다...")
history = agent.run_training(max_timesteps=300000)
print("학습 완료.")
