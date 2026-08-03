import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Normal, kl_divergence
import gymnasium as gym

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 표준편차는 학습 가능한 파라미터로 설정
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def log_prob(self, state, action):
        dist = self.forward(state)
        return dist.log_prob(action).sum(dim=-1)

    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.cpu().numpy()


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class TRPOAgent:
    def __init__(
        self,
        env_name,
        hidden_dim=64,
        gamma=0.99,
        lmbda=0.95,
        kl_delta=0.01,
        cg_iters=10,
        backtrack_coeff=0.5,
        backtrack_iters=10,
    ):
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_dim).to(device)
        self.value_net = ValueNet(obs_dim, hidden_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_delta = kl_delta  # Trust Region 크기
        self.cg_iters = cg_iters  # 켤레 기울기 반복 횟수
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters

    def get_action(self, state):
        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self.policy.get_action(state_t)
        return action[0]

    def compute_gae(self, rewards, values, masks):
        """
        GAE (Generalized Advantage Estimation) 계산
        """
        gae = 0
        returns = []
        values = torch.cat(
            [values, torch.zeros(1, 1).to(device)], dim=0
        )  # 마지막 상태 가치 0 처리
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.lmbda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return (
            torch.tensor(returns).to(device),
            torch.tensor(returns).to(device) - values[:-1].flatten(),
        )

    def surrogate_loss(self, new_policy, states, actions, advantages, old_log_probs):
        """
        TRPO의 목적 함수 (Surrogate Objective)
        L = E [ (pi_new / pi_old) * A ]
        """
        dist = new_policy(states)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def kl_divergence(self, new_policy, states, old_policy_dist):
        """
        평균 KL Divergence 계산
        """
        new_dist = new_policy(states)
        # detach()를 통해 oLd_poLicy는 상수로 취급
        return torch.distributions.kl_divergence(old_policy_dist, new_dist).mean()

    def hessian_vector_product(self, states, vector, old_policy_dist):
        """
        헤시안-벡터 곱 (Fisher Information Matrix * vector) 계산
        Pearlmutter's Trick 사용: Hx = grad(grad(KL) * x)
        """
        kl = self.kl_divergence(self.policy, states, old_policy_dist)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * vector).sum()
        grads_v = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_v])
        return flat_grad_grad_kl + 0.1 * vector  # Damping 추가

    def conjugate_gradient(self, states, b, old_policy_dist):
        """
        Ax = b 선형 방정식의 해 x를 근사적으로 구함 (A는 Fisher Matrix)
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.cg_iters):
            Ap = self.hessian_vector_product(states, p, old_policy_dist)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def line_search(
        self,
        states,
        actions,
        advantages,
        old_log_probs,
        old_policy_dist,
        full_step,
        expected_improve,
    ):
        """
        Backtracking Line Search
        조건 1: 목적 함수가 개선되어야 함
        조건 2: KL 제약 조건을 만족해야 함
        """
        old_params = torch.nn.utils.parameters_to_vector(self.policy.parameters())
        old_loss = self.surrogate_loss(
            self.policy, states, actions, advantages, old_log_probs
        )
        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff**i
            new_params = old_params + step_size * full_step
            torch.nn.utils.vector_to_parameters(new_params, self.policy.parameters())
            new_loss = self.surrogate_loss(
                self.policy, states, actions, advantages, old_log_probs
            )
            kl = self.kl_divergence(self.policy, states, old_policy_dist)
            # 목적함수(Reward)는 최대화해야 하므로 new_Loss > old_Loss 체크
            if new_loss > old_loss and kl < self.kl_delta:
                return True
        # 실패 시 원복
        torch.nn.utils.vector_to_parameters(old_params, self.policy.parameters())
        return False

    def train_step(self, max_steps=2048):
        states, actions, rewards, masks = [], [], [], []
        step_count = 0

        # 데이터 수집 (RoLLout)
        state, _ = self.env.reset()
        while step_count < max_steps:
            state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            dist = self.policy(state_t)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            masks.append(1 - float(done))

            state = next_state
            step_count += 1
            if done:
                state, _ = self.env.reset()

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)

        # GAE 계산
        with torch.no_grad():
            values = self.value_net(states)
        returns, advantages = self.compute_gae(rewards, values, masks)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )  # 정규화

        # PoLicy Update
        with torch.no_grad():
            old_policy_dist = self.policy(states)
            old_log_probs = old_policy_dist.log_prob(actions).sum(dim=-1)

        # 1. Gradient 계산 (Policy Gradient)
        loss = self.surrogate_loss(
            self.policy, states, actions, advantages, old_log_probs
        )
        grads = torch.autograd.grad(loss, self.policy.parameters())
        flat_grads = torch.cat([grad.view(-1) for grad in grads])

        # 2. Conjugate Gradient로 Search Direction 구하기 (H^-1 * g)
        step_dir = self.conjugate_gradient(states, flat_grads, old_policy_dist)

        # 3. 최대 스텝 크기 계산 (Lagrange MuLtipLier)
        shs = (
            step_dir * self.hessian_vector_product(states, step_dir, old_policy_dist)
        ).sum(0, keepdim=True)
        max_step_size = torch.sqrt(2 * self.kl_delta / (shs + 1e-8))[0]
        full_step = max_step_size * step_dir

        # 4. Line Search로 파라미터 업데이트
        expected_improve = (flat_grads * full_step).sum()
        self.line_search(
            states,
            actions,
            advantages,
            old_log_probs,
            old_policy_dist,
            full_step,
            expected_improve,
        )

        # VaLue Function Update (MSE Loss)
        for _ in range(10):  # VaLue net은 여러 번 업데이트
            values_pred = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values_pred, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        return sum(rewards) / (masks.count(0) + 1e-8)  # 평균 에피소드 보상


# TRPO 에이전트 생성 및 학습
agent = TRPOAgent("LunarLanderContinuous-v3", hidden_dim=64)
print("학습을 시작한다..")
total_steps = 0
max_timesteps = 100000
history = []

while total_steps < max_timesteps:
    avg_reward = agent.train_step(max_steps=2048)
    total_steps += 2048
    history.append(avg_reward)
    print(f"Steps: {total_steps}, Avg Reward: {avg_reward:.2f}")
print("학습 완료.")

# 학습 곡선 (Average Return)
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("TRPO Average Returns")
plt.xlabel("Updates")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# 비디오 저장 및 확인 함수
def save_video(agent, filename="go2_drl-17_trpo_book"):
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


# 비디오 저장 실행
save_video(agent)

# 일단 돌아는가고, 점수도 최소한 마이너스는 아닌데 착륙선이 착륙을 안하고 버티는 학습을 보여준다...
