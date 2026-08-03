import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}  |  PyTorch: {torch.__version__}")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 1. 운영체제별 폰트 설정
if platform.system() == "Windows":
    # 윈도우 맑은 고딕
    font_family = "Malgun Gothic"
elif platform.system() == "Darwin":
    # 맥 애플고딕
    font_family = "AppleGothic"
else:
    # 리눅스/코랩 나눔고딕
    font_family = "NanumGothic"

# 2. 폰트 설정
plt.rc("font", family=font_family)
# 3. 마이너스 부호 깨짐 해결
plt.rcParams["axes.unicode_minus"] = False


class DynaQ:
    """Dyna-Q: 실제 경험 + 모델 기반 시뮬레이션으로 Q-테이블을 학습한다."""

    def __init__(
        self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=10
    ):
        # 이건 Q 테이블 초기화겠지?
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # (s, a) -> (s', r, done)
        self.visited_sa = []  # 경험한 상태-행동 쌍
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.n_planning = epsilon, n_planning
        self.n_actions = n_actions

    def select_action(self, state):
        # ε 확률로 탐색
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # 아니면 최대 Q 값 인덱스가 행동
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        # ① Direct RL: 실제 전이로 Q-값 업데이트
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # ② Model Learning: 환경 모델(딕셔너리)에 전이(s', r, done) 저장
        self.model[(s, a)] = (s_next, r, done)
        if (s, a) not in self.visited_sa:
            self.visited_sa.append((s, a))

        # ③ Planning: 모델로 가상 경험 생성 → Q-값 추가 업데이트
        for _ in range(self.n_planning):
            # 경험치 없으면 그냥 나가고,
            if not self.visited_sa:
                break
            # 있으면 경험치들 중에서 랜덤 선택
            idx = np.random.randint(len(self.visited_sa))
            s_sim, a_sim = self.visited_sa[idx]
            # 내부 모델에서 s', r, done 꺼내서
            s_n, r_s, d_s = self.model[(s_sim, a_sim)]
            # 그걸로 Q 테이블 업데이트? 그럼 경험치가 너무 과대평가되지 않을까?
            tgt = r_s if d_s else r_s + self.gamma * np.max(self.Q[s_n])
            self.Q[s_sim, a_sim] += self.alpha * (tgt - self.Q[s_sim, a_sim])


class CliffWalkGrid:
    """4x12 그리드 클리프 워킹 환경"""

    def __init__(self):
        # 공간은 4x12, 출발점은 맨 왼쪽 아래, 도착점은 맨 오른쪽 아래
        self.rows, self.cols = 4, 12
        self.start, self.goal = (3, 0), (3, 11)
        # 그 사이의 맨 아래 격자들은 낭떠러지...
        self.cliff = [(3, c) for c in range(1, 11)]

    def reset(self):
        # 위치가 상태인데, 내부적으로는 가로 세로로 저장
        self.pos = self.start
        # 반환할 때는 좌상에서 우하로 가로 방향으로 순서대로 인덱스...
        return self.pos[0] * self.cols + self.pos[1]

    def step(self, action):
        r, c = self.pos
        if action == 0:
            # 북
            r = max(r - 1, 0)
        elif action == 1:
            # 남
            r = min(r + 1, self.rows - 1)
        elif action == 2:
            # 서
            c = max(c - 1, 0)
        elif action == 3:
            # 동
            c = min(c + 1, self.cols - 1)
        # 이동한 위치
        self.pos = (r, c)
        # 인덱스로 전환
        idx = r * self.cols + c
        # 낭떠러지에 떨어지면 -100점
        if self.pos in self.cliff:
            return self.reset(), -100, True
        # 목표에 도착하면 종료
        if self.pos == self.goal:
            return idx, 0, True
        # 아니면 매 스텝 -1점
        return idx, -1, False


# 실험 실행
planning_list = [0, 5, 20, 50]
results = {}
env = CliffWalkGrid()

# 과거 경험치를 몇 번 사용하느냐...
for n_plan in planning_list:
    # 상태 공간, 행동 공간 크기를 명시적으로 주고, 과거 경험치 사용 횟수도 줘서 에이전트 만들기...
    agent = DynaQ(48, 4, n_planning=n_plan)
    rewards = []
    # 120 스텝 제한으로 시뮬레이션
    for ep in range(120):
        s = env.reset()
        total = 0
        for _ in range(500):
            a = agent.select_action(s)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, s_next, done)
            total += r
            s = s_next
            if done:
                break
        rewards.append(total)
    results[n_plan] = rewards

# 시각화
plt.figure(figsize=(11, 5))
window = 10
for n_plan, rews in results.items():
    sm = np.convolve(rews, np.ones(window) / window, mode="valid")
    lbl = f"Planning={n_plan}" + (" (순수 Q-learning)" if n_plan == 0 else "")
    plt.plot(sm, label=lbl, linewidth=2)
plt.xlabel("에피소드")
plt.ylabel(f"보상 ({window}-에피소드 이동평균)")
plt.title("Dyna-Q: Planning 스텝 수에 따른 학습 속도", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("=> Planning 스텝이 많을수록 동일한 실제 경험에서 더 빠르게 학습한다!")

# 초기 학습 속도는 Planning 많을수록 좀 높아지는 듯도 하지만 수렴 안정성은 planning 적을수록 높은거 같은데...
