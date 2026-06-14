import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# 여긴 Hyperparameters가 왜 간단하지? 배치 크기는?
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # 경험치가 dqueue가 아니라 그냥 리스트...숫자가 많지 않다는 말인가?
        self.data = []

        # DQN보다 한 층이 적다...
        # 상태 s의 4개(카트 위치, 속도, 막대 각도, 각속도) 받아서 행동 2가지로 출력
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        # 옵티마이저도 그냥 망 안에 선언한다...
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # 첫 번째 층은 Relu, 두 번째는 확률로 나가야 하니 소프트맥스
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    # 훈련도 별도 함수로 빼는게 아니라 그냥 망 안에 설정한다...
    def train_net(self):
        # 총 보상 초기화하고
        R = 0
        self.optimizer.zero_grad()
        # 경험치를 마지막부터 역순으로 돌면서...
        for r, prob in self.data[::-1]:
            # 225쪽의 Gt, 손실 식인데...
            R = r + gamma * R
            loss = -torch.log(prob) * R
            # 한 번 계산할 때마다 역전파를 해야 하나? 이러면 미분값이 계속 더해지는데...
            # 사이토 고키 밑바닥 4권에서는 역전파를 for 루프 끝나고 했는데...마찬가지란 얘긴가?
            loss.backward()
        # 그래놓고 매개변수 갱신은 한 번에? 이것도 밑바닥 4권에서는 update 함수를 호출하는데...
        self.optimizer.step()
        self.data = []


def main():
    env = gym.make("CartPole-v1")
    # 훈련 대상인 정책망
    pi = Policy()
    score = 0.0
    print_interval = 20

    # 마찬가지로 만번 반복해서
    for n_epi in range(10000):
        # 초기 상태
        s, _ = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminate at 500 step.
            # 상태 s는 넘파이인 모양...토치 텐서로 순전파하면 소프트맥스로 행동 확률 나오고
            prob = pi(torch.from_numpy(s).float())
            # 결국 정규화해서 확률로 만들어준다는데, 그럼 위 순전파에서 소프트맥스를 계산할 필요가 있나? 음수를 없애려고?
            m = Categorical(prob)
            # 암튼 그 확률에 따라 어떤 행동 a를 선택...a는 prob의 확률 값이 아니라 인덱스 0 or 1...
            a = m.sample()
            # 환경에는 파이썬 숫자형으로 넣어줘야 하나?
            s_prime, r, done, truncated, info = env.step(a.item())
            # 정책망 데이터에 쌓는 경험치는 행동의 확률과 보상
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {}".format(n_epi, score / print_interval)
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
