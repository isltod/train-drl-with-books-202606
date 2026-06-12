import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        # 리스트와 비슷한 double ended queue - FIFO
        # 그냥 넣기만 해도 5만개 이후엔 맨 처음 들어왔던 데이터부터 버리면서 5만개를 유지
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        # 일단 n개의 경험 데이터를 받고
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        # 각각을 돌며 종류별로 담는데...
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            # 상태 s, s'은 그냥 리스트로 담고
            s_lst.append(s)
            # 행동 a와 보상 r, 종료 여부는 리스트 안에 리스트로 담는데...이 차이는?
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # 다 텐서로 바꿔서 반환
        return (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(np.array(a_lst)),
            torch.tensor(np.array(r_lst)),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_mask_lst)),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    """앞 장들의 Agent가 여기부터는 Net이 되는 모양..."""

    def __init__(self):
        super(Qnet, self).__init__()
        # 책의 아타리 그림과 달리 이건 다 완전 연결 계층으로...
        # 상태 s의 4개(카트 위치, 속도, 막대 각도, 각속도) 받아서 행동 2가지로 출력
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # 활성화 함수는 다 Relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # 관측값 받아서 엡실론 그리디로 행동 선택...학습시키는 에이전트에서 사용...
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    # 매번 학습은 10번만 돌아가나?
    for i in range(10):
        # memory = ReplayBuffer이고 s, a, r, s', done 32개를 텐서로 받아서
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # q는 학습 QNet 클래스, forward
        q_out = q(s)
        # (배치, 행동 종류) 텐서를 받아서, 그 중 행동 종류 축에서 a 인덱스로 선택된 값을 추려서 텐서로 만들기
        # 결론은 각 배치별 선택된 행동 가치 함수인가?
        q_a = q_out.gather(1, a)
        # q_target은 정답 QNet...순전파로 행동 가치 함수값
        # max는 각 열(1)에서 최댓값을 구해서 [0]으로 그 값(인덱스 말고) 반환...거기에 배치 차원 추가
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # 정답지로 쓸 행동 가치 함수 값 계산
        target = r + gamma * max_q_prime * done_mask
        # 손실 값은 목표망과 학습망의 행동 가치 함수 값의 차이인가...
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    # 훈련하는 학습망과 정답지로 쓸 목표망을 만들고...
    q = Qnet()
    q_target = Qnet()
    # 각 계층 매개변수들을 텐서로 매핑되는 딕셔너리 개체?
    # 학습망의 매개변수들을 목표망으로 이식시키는 코드인 듯...아래에서 20번에 한 번씩 다시 이식시킴...
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    # 옵티마이저는 학습망으로 만드네...
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # 만 번 돌면서
    for n_epi in range(10000):
        # 엡실론은 대략 8%에서 1%까지 줄여가면서 학습...
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        # 초기화 상태 받고
        s, _ = env.reset()
        done = False

        # 게임 끝날때까지 돌면서...
        while not done:
            # 학습 에이전트에게 상태와 엡실론 주고 엡실론 그리디로 행동 선택
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # 행동으로 다음 상태, 보상, 종료 여부 등을 받고
            s_prime, r, done, truncated, info = env.step(a)
            # train에서 종료된 것의 행동 가치 함수값이 0이 되도록 불리언을 0, 1로 바꾸고...
            done_mask = 0.0 if done else 1.0
            # 경험치를 누적...현재 보상은 값이 1 정도인거 같은데 커서 학습이 안된다? 그래서 100으로 나눈다...
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            # 다음을 위해 s'을 s로 바꾸고 반복
            s = s_prime

            score += r
            if done:
                break

        # 경험을 2천번 이상 쌓으면 훈련을 하는데...
        if memory.size() > 2000:
            # 여기서 학습망 q의 매개변수들을 학습시키고..
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            # 그런 에피소드를 200번 할 때마다 목표망의 매개변수를 학습망과 같게 만들고...
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
