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
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 이게 C51 네트워크라고...
class DistributionalDQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, n_atoms):
        """
        Distributional DQN 네트워크 초기화
        :param n_atoms: 분포를 표현할 지지점(Support)의 개수 (보통 51개 사용 -> C51)
        """
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        # 특징 추출 레이어
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 출력 레이어: (행동 수 * 원자 수) 크기로 출력
        # 여기 원자 수가 고정 support 수란 얘기일테고, 그럼 51개?
        self.fc_out = nn.Linear(hidden_size, n_actions * n_atoms)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature_layer(x.float())
        out = self.fc_out(features)

        # 구조 변경: (Actions와 Atoms 차원 분리...)
        out = out.view(batch_size, self.n_actions, self.n_atoms)

        # 소프트맥스에 로그를 취한 로그 확률 분포(Log Probability)
        # Wasserstein 대신 KL divergence를 사용하면서 손실함수가 크로스 엔트로피가 되는데,
        # 크로스 엔트로피에서 로그 확률을 이용하니까 그걸 출력하는건가?
        return F.log_softmax(out, dim=2)


# 재생 버퍼도 단순 버전이고...
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# 학습 클래스...이게 에이전트인가?
class PytorchWrapper:
    def __init__(
        self,
        env_name,
        hidden_size=128,
        lr=1e-3,
        capacity=100000,
        gamma=0.99,
        batch_size=64,
        sync_rate=10,
        n_atoms=51,
        v_min=-10,
        v_max=10,
    ):
        self.env_name = env_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_rate = sync_rate

        # C51 하이퍼파라미터
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        # Support vector 생성 (z_i): v_min에서 v_max까지 등간격 벡터
        # 이건 전체 과정 중에 절대 변하기 않고,
        # 다음 단계의 확률들이 나오면 그걸 이 고정 막대에 비율대로 붙여가며 업데이트 한다...
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # 환경 생성
        self.env = gym.make(env_name, render_mode="rgb_array")
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # 네트워크 초기화
        self.q_net = DistributionalDQN(obs_size, hidden_size, n_actions, n_atoms).to(
            device
        )
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity)

    def get_action(self, state, epsilon):
        """
        행동 선택: 분포의 기대값(Mean)을 계산하여 Q값이 가장 큰 행동을 선택한다.
        Q(s, a) = sum(z_i * p_i)
        """
        # 행동 선택 방식은 일단 ε-greedy 방식이고...
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            # 그 중 greedy 방식에서 달라지는데...
            state_t = torch.tensor(np.array([state]), device=device)
            with torch.no_grad():
                # 굳이 Log probability를 받아서 Probability로 바꾸네? 왜 이러지?
                log_probs = self.q_net(state_t)
                probs = log_probs.exp()

                # 기대값 계산, * 는 원소별 곱
                # (Batch 1, Action 4, Atoms 51) * (Atoms 51,) -> (Batch, Action, Atoms).sum -> (Batch, Action)
                q_values = (probs * self.support).sum(dim=2)

            # 기대값 중 최대값이 인덱스가 행동...
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 일단 기존과 마찬가지로 재생 버퍼에서 샘플링 해서 s, a, r, done, s' 별로 텐서로 묶는데...
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(np.array(states), device=device)
        # 이건 인덱싱을 위해 LongTensor로 만든다는데...그게 뭔지...
        actions = torch.tensor(actions, device=device).long()
        rewards = torch.tensor(rewards, device=device).float()
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), device=device)
        # 배치 사이즈는 정해주는 건데 왜 여기서 따로 설정하지?
        batch_size = states.size(0)

        # --- 1. 다음 상태의 행동 선택 및 타겟 분포 계산 (Categorical Algorithm) ---
        with torch.no_grad():
            # 이것도 s'의 Log 확률 분포를 얻고 그걸 굳이 다시 확률로 바꾸는데...왜 이렇게 하지?
            next_log_probs = self.target_q_net(next_states)  # (Batch, Actions, Atoms)
            next_probs = next_log_probs.exp()

            # 다음 상태의 행동 선택 (Q값 = 기대값 기준 Greedy) - 근데 이걸 왜 그냥 get_action으로 구하질 않지?
            next_q_values = (next_probs * self.support).sum(dim=2)
            next_actions = next_q_values.argmax(dim=1)  # (Batch,)

            # 다음 행동으로 선택된 다음 상태의 분포(pi 들)만 가져오기
            # next_probs[i, next_actions[i]] 반복을 효율적으로 수행
            next_dist = next_probs[range(batch_size), next_actions]  # (Batch, Atoms)

            # 여기서부터 타겟 분포 투영 절차 - 가까운 support 들에 비율대로 나눠주기...
            # support 자체가 Q 값들...target = r + γQ'의 distributional 버전...
            # support는 계속 고정 상태고, 다음 Z' 계산 자체가 고정된 support에 γ를 곱하고 r을 더하고,
            # 그걸 다시 이 고정 support로 프로젝션 하는 방식으로 진행된다...
            t_z = rewards.unsqueeze(1) + (
                1 - dones.unsqueeze(1)
            ) * self.gamma * self.support.unsqueeze(0)
            # r + γZ'으로 옮겨진 z들의 범위를 제한하고
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # 인덱스 계산 (bj = (현재 위치 - 최소 위치) / 한칸 크기)
            b = (t_z - self.v_min) / self.delta_z
            # 4.3칸이라고 나오면 왼쪽이 4, 오른쪽이 5번 support (배치 64, atom 51)
            l = b.floor().long()
            u = b.ceil().long()

            # 이게 최종적으로 타겟 분포가 될 벡터인데...프로젝션 위해서 일단 0으로 초기화
            target_dist = torch.zeros_like(next_dist)

            # 아래서 (배치, atom) 차원을 (배치 x atom)차원으로 펼쳐서 연산을 하려고 하니,
            # 배치 0은 인덱스가 0부터 시작하면 되지만, 배치 1은 인덱스가 51부터 시작하는 등, 뒤로 가야 한다...
            # 그래서 [0, 51, 102,...] 이런 모양의 텐서를 offset이란 이름으로 만들어둔다...
            offset = (
                torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size)
                .long()
                .unsqueeze(1)
                .to(device)
            )

            # l과 u 인덱스에 확률을 나눠서 할당 (Linear interpolation)
            aa = torch.tensor([[1, 2], [3, 4]])
            bb = aa.view(-1)
            # 1. view(-1) - 1차원 텐서로 펼치기,
            # 2. (l + offset) - 배치 1 단계마다 51, 102,...를 더한 텐서 - (배치, atom)을 1차원으로 펼쳐서 계산하기 위한 인덱스...
            # 3. (u.float() - b) - 오른쪽 거리 0.7
            # 4. index_add_ - 0번 차원에, 2번 인덱스 위치에, 3번 값을 더한다...
            target_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # --- 2. 현재 상태의 예측 분포 계산 ---
        current_log_probs = self.q_net(states)  # (Batch, Actions, Atoms)
        # 현재 수행한 행동에 대한 Log 분포만 선택 (Batch, Atoms)
        current_log_dist = current_log_probs[range(batch_size), actions]

        # --- 3. 손실 함수 계산 (KL Divergence) ---
        # Cross Entropy: - sum(target * log(prediction))
        # 역시나 여기서 로그 확률을 그대로 사용하려고 하는 모양인데...앞에서는 다시 exp도 사용하는데, 정말 이게 더 좋을까?
        loss = -(target_dist * current_log_dist).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_training(self, max_episodes=600, max_steps=400):
        total_rewards = []

        # 에피소드 수만큼 돌면서 경험 저장하고 학습하는데...여긴 간단하게 하네...
        for episode in tqdm(range(max_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0

            epsilon = max(0.01, 1.0 - (episode / 200))

            for step in range(max_steps):
                # 어쨌거나 action은 기존과 마찬가지로 스칼라로 받으니까...
                action = self.get_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.append((state, action, reward, done, next_state))
                state = next_state
                episode_reward += reward

                self.train_step()

                if done:
                    break

            if episode % self.sync_rate == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            total_rewards.append(episode_reward)

            if episode % 20 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

        return total_rewards

    def save_video(self, filename="dist_dqn_video"):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=filename)

        state, _ = env.reset()
        done = False
        while not done:
            action = self.get_action(state, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


# C51 모델 생성
# v_min, v_max는 환경의 보상 범위를 고려하여 설정해야 함 (LunarLander는 -200 ~ 200 정도가 적당하나 여유있게 잡음)
agent = PytorchWrapper(
    "LunarLander-v3", hidden_size=128, lr=1e-3, n_atoms=51, v_min=-10, v_max=100
)
# v_min/max 조절이 중요하다고 하는데, 이걸 어떻게 해야 하는지도 모르겠고, 이걸 안해도 되는 QR이나 IQN도 있잖아?

# 학습 시작
print("Distributional DQN (C51) 학습을 시작한다...")
history = agent.run_training(max_episodes=600)
print("학습 완료.")

# 결과 시각화 - 학습 곡선
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Distributional DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

agent.save_video("go2_drl-07_c51")

# 생각외로 이것도 잘 하는거 같다...
