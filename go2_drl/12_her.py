import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 행위자 네트워크 - 구조는 SAC + HER
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size, action_dim, max_action):
        """
        Actor: (상태 + 목표) -> 행동 분포 (Mean, Std)
        """
        super().__init__()
        self.max_action = float(max_action)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state_goal):
        # 이거만 달라지는데, (상태 concat 목표)가 벡터로 들어오는 모양...
        x = self.net(state_goal)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 안정적인 학습을 위해서 표준편차 범위를 -20~2로 제한한다고...
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        # 행동을 그냥 Normal에서 샘플링하면 역전파를 전달할 수 없으니,
        # Reparameterization 트릭이란 걸 써서 그게 가능하도록...μ + σ*ε
        z = dist.rsample()
        action = torch.tanh(z)

        # 이건 왜 하는지 잘 모르겠는데...Tanh 변환에 따른 보정항 추가해서 Log Probability 계산
        # log_prob = log_prob_normal - log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 행위자 순전파는 행동의 평균과 표준편차가 아니라, 행동과 로그확률을 반환...
        return action * self.max_action, log_prob


# 비평가 네트워크
class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, action_dim):
        """
        Critic: (상태 + 목표) + 행동 -> Q값
        """
        super().__init__()

        # Q1
        self.net1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Q2
        self.net2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_goal, action):
        # 계산 효율은 때문인지, 굳이 (s concat g)에 a를 붙여서 처리...
        x = torch.cat([state_goal, action], dim=1)
        # 그냥 같은 입력에 두 네트워크가 각각 계산한걸 반환한다...
        return self.net1(x), self.net2(x)


# HER 재생 버퍼...HER 알고리즘에서 제일 중요한 부분...
# 특히 배치 샘플링이 중요한데, her_ratio로 원래 목표 대신에
# 그 에피소드 내에서 실제로 도달했던 아무 지점을 목표로 대체(Future Strategy)해서 보상을 재계산한다고...
class HERReplayBuffer:
    def __init__(self, capacity, env, her_ratio=0.8):
        self.buffer = deque(maxlen=capacity)
        self.env = env
        self.her_ratio = her_ratio  # Goal을 바꿀 확률 (미래 시점의 달성 상태로)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode_trajectory):
        """한 스텝의 (s, a, r, done, s')이 아니라 에피소드 전체 궤적을 저장"""
        self.buffer.append(episode_trajectory)

    def sample(self, batch_size):
        # 1. 에피소드 인덱스 선택 - 현재 저장된 모든 경험 중 batch_size 만큼 랜덤 선택
        indices = np.random.randint(0, len(self.buffer), batch_size)

        # s, a, s, g, g', r을 메인 dequeue가 아니라 별도의 리스트로 만들어서 사용...
        states, actions, next_states = [], [], []
        desired_goals, achieved_goals = [], []
        rewards = []

        # 각 에피소드 궤적별로 반복해서
        for idx in indices:
            # 실제 궤적 내용을 받고
            episode = self.buffer[idx]

            # 2. 에피소드 내에서 다시 무작위로 현재 타임스텝(t) 선택해서
            t = np.random.randint(0, len(episode))
            transition = episode[t]
            # 그 전이에서 s, a, s', g 선택
            obs = transition["obs"]
            action = transition["action"]
            next_obs = transition["next_obs"]
            goal = transition["desired_goal"]

            # 3. her_ratio 확률로 HER 적용 여부 결정 (Future Strategy)
            if np.random.random() < self.her_ratio:
                # HER 적용이면, 현재 t에서 그 궤적 끝 사이에 특정 미래를 무작위 선택
                future_t = np.random.randint(t, len(episode))
                future_transition = episode[future_t]
                # 해당 미래 로봇팔 위치가 achieved_goal
                goal = future_transition["achieved_goal"]

            # 4. 보상 재계산 - HER는 play 중에 즉각 보상을 받지 않고 이렇게 나중에 HER 적용하고 계산하는 모양...
            # unwrapped는 모든 Wrapper 층 제거
            # compute_reward: achieved_goal, desired_goal, info 받아서 보상을 재계산 해주는 메서드라고...
            # 여기 goal은 HER 적용 안되면 desired_goal, 적용되면 achieved_goal인 상태,
            # info는 중요하지 않은 듯...원래는 desired_goal 달성 여부가 반환되는데 빈 사전으로 사용하네...
            reward = self.env.unwrapped.compute_reward(
                transition["achieved_goal"], goal, {}
            )

            # s, a, s', g(or g'), r'을 리스트에 추가...
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            desired_goals.append(goal)
            rewards.append(reward)  # <--- 계산된 보상을 리스트에 추가!

        # 배열 변환 및 결합
        states = np.array(states)
        next_states = np.array(next_states)
        desired_goals = np.array(desired_goals)
        actions = np.array(actions)
        # 이렇게 해서 에러를 잡았다는 메모가 붙어있는데...원래가 (256, 1) 형태라 있으나 마나인데?
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.zeros_like(rewards)

        # 상태와 원래 목표는 concat 해서 사용하는 입력 벡터로 구성
        inp_states = np.concatenate([states, desired_goals], axis=1)
        inp_next_states = np.concatenate([next_states, desired_goals], axis=1)

        return inp_states, actions, rewards, dones, inp_next_states


# 학습 클래스
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=256,
        lr=1e-3,
        capacity=1000000,
        gamma=0.98,
        batch_size=256,
        tau=0.005,  # Soft update ratio
        alpha=0.2,  # Entropy coefficient (Temperature)
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha

        # 환경 생성 (gymnasium-robotics)
        self.env = gym.make(env_name, render_mode="rgb_array")

        # 상태 및 행동 차원 확인 - HER에서는 상태가 observation...
        # observation_space는 사전 형태로 키는 'observation', 'desired_goal', 'achieved_goal'
        # 별다른 처리 안했는데 그냥 gymnasium-robotics에서 만들어 준 환경이 HER 구조를 가지네...
        obs_dim = self.env.observation_space["observation"].shape[0]
        goal_dim = self.env.observation_space["desired_goal"].shape[0]
        input_dim = obs_dim + goal_dim  # 네트워크 입력은 상태 + 목표

        # shape이 (4,) 모양이므로 그 중 4를 정수로 받기
        action_dim = self.env.action_space.shape[0]
        # action 차원마다 high 값이 하나 씩 있어서 [h1, h2, h3, h4] 형태
        max_action = self.env.action_space.high[0]

        # 네트워크 초기화
        self.actor = Actor(input_dim, hidden_size, action_dim, max_action).to(device)
        self.critic = SoftQNetwork(input_dim, hidden_size, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # 최적화기
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # HER 버퍼
        self.buffer = HERReplayBuffer(capacity, self.env)

    def get_action(self, obs, goal):
        """행동 선택"""
        # 상태(obs)와 목표를 결합해서 그걸 상태로 이용...
        inp = np.concatenate([obs, goal])
        state_t = torch.tensor(np.array([inp]), dtype=torch.float32, device=device)

        # 학습에는 순전파로 역전파 전달하고, 테스트 등에서는 get_action으로 역전파 없이 행동 샘플링
        with torch.no_grad():
            action, _ = self.actor(state_t)

        # 이것도 배치 차원 없애고 (2,) 형태로 만들기...
        return action.cpu().numpy()[0]

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    # 하나의 에피소드 궤적을 생성해서 저장...
    def play_episode(self):
        """
        에피소드를 실행하고 궤적(Trajectory)을 저장
        """
        # HER은 딕셔너리 받아서 거기서 상태는 observation, 원래 목표는 desired_goal...
        obs_dict, _ = self.env.reset()
        obs = obs_dict["observation"]
        # 원래 목표는 절대 안바뀌는 모양...여기서 한 번 정하고 그대로 쓴다...로봇 팔 끝이 도달해야 하는 원래 목표 x, y, z
        desired_goal = obs_dict["desired_goal"]

        # 에피소드 궤적은 {s, a, s', g, g'} 사전들의 리스트...
        episode_trajectory = []
        done = False
        truncated = False
        episode_reward = 0

        # 에피소드 끝까지 돌면서...
        while not (done or truncated):
            # 상태와 원래 목표를 concat하고, 순전파 아닌 get_action으로 역전파 연결 없이 행동 샘플링
            # ndarray(4,) 형태로, 손끝의 좌표 변위 dx, dy, dz 세 개와 그리퍼 열고 닫기 정도
            action = self.get_action(obs, desired_goal)
            # step의 반환도 dict, r, done, truncated, info 형태로 다르고...
            next_obs_dict, reward, done, truncated, info = self.env.step(action)

            # 상태는 집게 위치 3, 집게 속도 3, 집게 벌어짐 2, 손가락 속도 2로 (10,) 구조...
            next_obs = next_obs_dict["observation"]
            # 현재 달성한 상태 (HER에서 중요) - 로봇 팔 끝단의 현재 x, y, z
            achieved_goal = next_obs_dict["achieved_goal"]

            # 궤적 저장 (obs, action, next_obs, desired_goal, achieved_goal) - 특이하게 즉각 보상이 없다...나증에 계산...
            episode_trajectory.append(
                {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "desired_goal": desired_goal,
                    "achieved_goal": achieved_goal,
                }
            )

            obs = next_obs
            episode_reward += reward

        # 에피소드 전체를 버퍼에 저장 - 실제로는 아래 반환값보다 이걸 이용하네...
        self.buffer.append(episode_trajectory)

        # 사용하건 말건 어쨌거나 에피소드의 즉각보상 누적과 성공 여부를 반환...
        return episode_reward, info["is_success"]

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        # HER 버퍼에서 샘플링 (여기서 목표 교체 및 보상 재계산이 일어남)
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)

        # SAC Update Logic
        # 1. Critic Update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            y_target = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, y_target) + F.mse_loss(
            current_q2, y_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor Update
        new_actions, log_probs = self.actor(states)
        q1, q2 = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft Update
        self.soft_update(self.critic, self.critic_target)

    def run_training(self, max_epochs=200, episodes_per_epoch=10, updates_per_epoch=40):
        success_rates = []

        for epoch in range(max_epochs):
            # 1. 에피소드 수집
            total_success = 0
            for _ in range(episodes_per_epoch):
                _, is_success = self.play_episode()
                total_success += is_success

            # 2. 학습 수행
            for _ in range(updates_per_epoch):
                self.train_step()

            success_rate = total_success / episodes_per_epoch
            success_rates.append(success_rate)

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Success Rate: {success_rate:.2f}")

        return success_rates

    def save_video(self, filename="her_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        obs_dict, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = self.get_action(obs_dict["observation"], obs_dict["desired_goal"])
            obs_dict, _, done, truncated, _ = env.step(action)
        env.close()


# SAC + HER 모델 생성
# FetchReach는 비교적 쉬운 환경이므로 epoch를 적게 설정해도 금방 100%에 도달한다.
agent = PytorchWrapper("FetchReach-v4", hidden_size=256, lr=1e-3, batch_size=256)

# 학습 시작
print("SAC + HER 학습을 시작한다...")
history = agent.run_training(
    max_epochs=1000, episodes_per_epoch=10, updates_per_epoch=40
)
print("학습 완료.")
