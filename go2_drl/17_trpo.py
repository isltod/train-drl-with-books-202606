"""
실습 코드라고 받은 노트북 파일의 코드인데, 중간에 flat_param 문제가 있어 실행시킬 수가 없다.
뭘 하는 함수인지 짐작으로 고쳐볼 수는 있겠지만 TRPO 알고리즘에 대한 이해가 없어서 그러기도 힘들다.
대신 책의 소스코드는 이것과 다르고 실행되는 거 같다.
17_trpo_book 참고...
"""

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


# 수학적 유틸리티 함수들이라고...
def flat_grad(grads, params):
    """
    기울기 튜플을 받아 그래프를 유지하며 1차원 벡터로 평탄화합니다.
    .data를 사용하면 2차 미분이 불가능해지므로 주의해야 합니다.
    """
    grad_flatten = []
    for grad, p in zip(grads, params):
        if grad is None:
            # 기울기가 없는 경우는 매개변수 차원 수만큼 0 텐서..numel()은 모든 차원 곱 반환...
            grad_flatten.append(torch.zeros(p.numel(), device=device))
        else:
            # .data를 쓰지 않고 view만 사용하여 그래프 연결을 유지합니다.
            grad_flatten.append(grad.reshape(-1))
    # 그렇게 만든 기울기 텐서들의 리스트를 이어붙여서 반환...
    return torch.cat(grad_flatten)


def update_model(model, new_params):
    """평탄화된 파라미터를 다시 모델에 주입"""
    start_idx = 0
    # 모델의 매개변수 텐서마다 돌면서...
    for param in model.parameters():
        # 매개변수 텐서 원소 수로,
        param_length = param.numel()
        # 순서대로 인덱스 증가시키면서 평탄화된 텐서에서 매개변수 받아서
        new_param = new_params[start_idx : start_idx + param_length]
        # 이건 뭐 하는거? 뭔가 shape을 맞춰주는거 같은데...
        new_param = new_param.view(param.size())
        param.data.copy_(new_param)
        # 다음 매개변수 텐서 변환을 위해서 시작 인덱스 이동
        start_idx += param_length


# 이건 더 모르겠네...나중에 호출할 때 보자...------------------------------------------------------------
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    켤레 기울기법 (Conjugate Gradient Method)
    Ax = b 에서 x를 근사적으로 구함 (여기서 A는 Fisher Information Matrix)
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.dot(p, z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr

        if rdotr < residual_tol:
            break
    return x


# 행위자 네트워크 - 이것들은 간단한데...
class Actor(nn.Module):
    def __init__(self, obs_size, hidden_size, action_dim):
        super().__init__()
        # 일단 구조는 fc, tanh, fc, tanh, 마지막에 μ, logσ
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = self.net(x)
        # 평균은 그냥
        mean = self.mean_layer(x)
        # 표준편차는 log 없애고(이럴러면 왜 log라고?),
        # expand_as는 새로 메모리를 만들지 않고 뷰만 늘려서 mean과 같은 크기로 보이게 만들기..
        # 대신 크기가 1인 차원만 늘릴 수 있다고...
        # 근데 여긴 std와 mean이 shape이 같아서 expand_as 왜 하는지 모르겠다...
        std = self.log_std.exp()
        std = self.log_std.exp().expand_as(mean)
        return mean, std


# 비평가 네트워크
class Critic(nn.Module):
    def __init__(self, obs_size, hidden_size):
        super().__init__()
        # 비평가는 마지막 μ, logσ 없는 행위자 구조...
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# 학습 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=64,
        gamma=0.99,
        gae_lambda=0.95,  # GAE 파라미터
        delta=0.01,  # KL Divergence 제약 상수 (Trust Region 크기)
        damping=0.1,  # 수치 안정성을 위한 댐핑 계수
        cg_iters=10,  # Conjugate Gradient 반복 횟수
        backtrack_iters=10,  # 라인 서치 반복 횟수
        backtrack_coeff=0.8,  # 라인 서치 감쇠 비율
        n_steps=2048,
    ):

        self.env_name = env_name
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.delta = delta
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.n_steps = n_steps

        # 환경 생성 - Luna Lander Continuous v3, 상태 8, 행동 2
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # 네트워크 생성
        self.actor = Actor(obs_size, hidden_size, action_dim).to(device)
        self.critic = Critic(obs_size, hidden_size).to(device)

        # Critic용 최적화기 (Actor는 Optimizer 안쓰고 TRPO 로직으로 직접 업데이트 - 그래서 더 복잡...)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def get_action(self, state):
        # 역전파 없는 테스트 행동은 간단하게 행위자 순전파 평균으로...
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _ = self.actor(state_t)
        return mean.cpu().numpy()[0]

    def compute_gae(self, rewards, values, dones, next_value):
        """
        GAE (Generalized Advantage Estimation) 계산
        A = δ + γ* λ * A'
        이게 λ가 0이면 TD고 1이면 MC라는데..왜 그런거냐...암튼 현재는 0.95
        """
        # Advantage 추정값을 0으로 초기화하고...
        advantages = torch.zeros_like(rewards).to(device)
        # λ 추정치도 0으로 초기화
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
        returns = advantages + values
        # 반환은 t 별로 GAE 추정과 Q
        return advantages, returns

    def fisher_vector_product(self, vector, states):
        """
        Fisher Information Matrix와 벡터의 곱 (Hv) 계산
        """
        # 1. 먼저 행동 분포를 미분 가능하도록 순전파로 얻고
        mean, std = self.actor(states)
        dist = Normal(mean, std)

        # 2. 고정된 Old Policy로 넣어서 그래프에서 분리...근데 현재 mean = mean_old인데...
        with torch.no_grad():
            mean_old, std_old = mean.detach(), std.detach()
            dist_old = Normal(mean_old, std_old)

        # 3. KL Divergence 계산 - 근데 dist_old == dist 일텐데?
        kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean()

        # 4. 1차 미분 (create_graph=True - 고차 미분 가능)
        # torch.autograd.grad - 기울기를 .grad 속성에 누적하지 않고 바로 반환하는 함수...
        # 여기서 grads는 텐서들의 튜플이며, 각 텐서는 grad_fn을 가집니다.
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)

        # 수정된 flat_grad 호출
        flat_grads = flat_grad(grads, self.actor.parameters())

        # 5. KL_grad * vector 내적
        # vector는 CG에서 온 상수 벡터이므로 detach 상태여도 무방합니다.
        kl_v = (flat_grads * vector).sum()

        # 6. 2차 미분 (이제 kl_v에 grad_fn이 살아있어 에러가 나지 않습니다)
        grads_2nd = torch.autograd.grad(kl_v, self.actor.parameters())
        flat_grads_2nd = flat_grad(grads_2nd, self.actor.parameters())

        return flat_grads_2nd + self.damping * vector

    def train_step(self):
        # 1. 데이터 수집 (Rollout)
        states, actions, rewards, dones, values = [], [], [], [], []

        state, _ = self.env.reset()

        # n step 단위로 묶어서 실행/학습...
        for _ in range(self.n_steps):
            state_t = torch.tensor(
                np.array([state]), dtype=torch.float32, device=device
            )

            # 데이터 수집 단계는 역전파 끊고 a, V를 구하나?
            with torch.no_grad():
                # 행동은 평균과 표준편차로 정규분포에서 샘플링
                mean, std = self.actor(state_t)
                dist = Normal(mean, std)
                action = dist.sample()
                # 상태 가치는 비평가 순전파
                value = self.critic(state_t)

            # 행동으로 s', r, 종료여부 얻고
            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done_flag = terminated or truncated

            # n step 단위로 묶어준다...
            states.append(state_t)
            actions.append(action)
            rewards.append(reward)
            dones.append(done_flag)
            values.append(value)

            state = next_state
            # n step 중에 종료되면 게임 다시 시작
            if done_flag:
                state, _ = self.env.reset()

        # s, a, r, done, V는 n step 단위로 텐서로 묶어서...
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        values = torch.cat(values).squeeze()

        # n step 바로 다음 상태 가치 계산 (GAE용)
        with torch.no_grad():
            # 여기 state는 위 for 문 마지막의 state = next_state 때문에 n step 바로 다음 상태를 가리킨다...
            next_state_t = torch.tensor(
                np.array([state]),
                dtype=torch.float32,
                device=device,
            )
            next_value = self.critic(next_state_t).squeeze()

        # GAE 계산
        # 각 t 별로 이어진 텐서로, r: 즉각 보상들, v: 비평가 평가 상태 가치, d: 종료 여부,
        # v'은 n step 다음 t'의 상태 가치 스칼라
        # 반환 값은 시간 t 별로 묶은 GAE 추정치들과 Q 값들...
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        # Advantage 정규화 - 단지 학습 안정화를 위해서 취한다고...
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Critic 업데이트 (MSE Loss)
        for _ in range(10):  # Critic은 여러 번 업데이트
            # 현재 비평가가 예측하는 상태 가치? Q?
            v_pred = self.critic(states).squeeze()
            # returns는 A + V인데 그걸 비평가 V와 비교? 비평가가 Q를 학습하나?
            v_loss = F.mse_loss(v_pred, returns)
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()

        # 3. Actor 업데이트 (TRPO)

        # 현재 정책의 Log Probability 계산 - 근데 현재를 자꾸 old라고 해놓네....
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        old_log_probs = dist.log_prob(actions).sum(dim=-1).detach()

        # Surrogate Loss 함수 정의
        def get_loss(volatile=False):
            # 휘발성 플래그 지정하면
            if volatile:
                # 역전파 끊고 mean, std, log_probs 계산하고...
                with torch.no_grad():
                    mean, std = self.actor(states)
                    dist = Normal(mean, std)
                    # 근데 이러면 위에 old_log_probs와 여기 log_probs가 차이가 뭐지?
                    log_probs = dist.log_prob(actions).sum(dim=-1)
            else:
                # 휘발성 아니면 역전파 연결 상태에서 mean, std, log_probs 계산해서...
                mean, std = self.actor(states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)

            # 두 확률 비율에 Advantage 들 곱한 값들의 평균을 반환...
            aa = log_probs.cpu().detach().numpy()
            bb = old_log_probs.cpu().detach().numpy()
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        # 1차 미분 (Policy Gradient) 계산
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = flat_grad(grads, self.actor.parameters())

        # Conjugate Gradient로 Search Direction(x) 계산: Hx = g
        # 여기서 Fisher Vector Product 함수를 인자로 넘김
        fvp = lambda v: self.fisher_vector_product(v, states)
        step_dir = conjugate_gradient(fvp, -loss_grad, self.cg_iters)

        # Step Size 계산 (Lagrange Multiplier)
        # beta = sqrt(2 * delta / x^T H x)
        shs = 0.5 * (step_dir * fvp(step_dir)).sum(0, keepdim=True)
        max_step = torch.sqrt(self.delta / shs[0])
        full_step = step_dir * max_step

        # Line Search (Backtracking)
        # 이게 뭔 함수인지 모르겠고, 지금으로선 추측도 못하겠다...그리고 여기 코드는 책과 또 틀리다...
        old_params = flat_params(self.actor)
        expected_improve = -(loss_grad * full_step).sum()

        flag = False
        for i in range(self.backtrack_iters):
            # Step 크기를 줄여가며 시도
            step_frac = self.backtrack_coeff**i
            new_params = old_params + step_frac * full_step
            update_model(self.actor, new_params)

            # 조건 확인: Loss가 개선되었는가? KL 제약을 만족하는가?
            new_loss = get_loss(volatile=True)

            # KL 계산
            with torch.no_grad():
                mean_new, std_new = self.actor(states)
                dist_new = Normal(mean_new, std_new)
                mean_old, std_old = mean.detach(), std.detach()
                dist_old = Normal(mean_old, std_old)
                kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()

            # Actual improvement > 0 (Minimizing negative loss)
            loss_improve = loss - new_loss

            if kl <= self.delta * 1.5 and loss_improve > 0:
                flag = True
                break

        # 조건을 만족하지 못하면 업데이트 취소
        if not flag:
            update_model(self.actor, old_params)

        return returns.mean().item()

    def run_training(self, max_timesteps=200000):
        total_steps = 0
        rewards_history = []

        while total_steps < max_timesteps:
            mean_return = self.train_step()
            total_steps += self.n_steps
            rewards_history.append(mean_return)

            if total_steps % (self.n_steps * 5) == 0:
                print(f"Steps: {total_steps}, Mean Return: {mean_return:.2f}")

        return rewards_history

    def save_video(self, filename="trpo_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False

        while not done:
            action = self.get_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# TRPO 모델 생성
agent = PytorchWrapper(
    "LunarLanderContinuous-v3",
    hidden_size=128,
    gamma=0.99,
    delta=0.01,  # KL Constraint
    n_steps=2048,
)

# 학습 시작
print("TRPO (Trust Region Policy Optimization) 학습을 시작한다...")
history = agent.run_training(max_timesteps=200000)
print("학습 완료.")
