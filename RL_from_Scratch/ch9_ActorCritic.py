import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
# 종료까지 기다리지 않고 10개만 모이면 평가한다...
n_rollout = 10


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        # 하나의 망에 행동과 상태 가치 층을 다 둔다? 이러면 fc1은 pi 역전파와 v 역전파에서 모두 업데이트? 합쳐지나?
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # 행동 벡터를 내놓는 순전파 같은데...
    def pi(self, x, softmax_dim=0):
        # 입력 층? 활성화 거치고
        x = F.relu(self.fc1(x))
        # 그걸 pi용 층을 통과시키고 소프트맥스 걸어서 행동 벡터로 반환...
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # 이건 상태 가치를 내놓는 순전파?
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    # 경험치를 data에 저장..
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        # 각 경험을 끝까지(10번?) 돌면서
        for transition in self.data:
            # 각 경험의 상태, 행동, 다음 상태, 종료여부 받고
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            # 이것도 학습 잘 시키려 나누는건가?
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            # 이것도 종료되면 보상 값을 이용하지 않으려고 0으로 마스킹
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        # s, a, r, s', done을 텐서로 만들어서 반환
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_lst, dtype=torch.float),
        )
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        # 경험 10개를 묶어 s, a, r, s', done 텐서로 받고
        s, a, r, s_prime, done = self.make_batch()
        # 다음 상태 가치...이게 타겟이다..
        td_target = r + gamma * self.v(s_prime) * done
        # 그걸 현재 가치에서 빼면 그게 어드벤티지(델타)...
        delta = td_target - self.v(s)

        # (배치, 행동 종류) 벡터가 나올 것이고,
        pi = self.pi(s, softmax_dim=1)
        # 선택 행동 값만 추린 텐서를 만들고...
        pi_a = pi.gather(1, a)
        # 앞단은 정책 손실..뒤는 상태 가치 손실이라고...
        # detach는 상수 취급이라...연산들을 떼어낸다...왜지? 마이너스는 경사하강법에 맞추기 위해서라고...
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(
            # 여기 detach도 상수 취급이라는데...v(s)만 변하고 타겟은 안 변하게 한다고...
            self.v(s),
            td_target.detach(),
        )

        self.optimizer.zero_grad()
        # 그리고 손실이 아니라 손실 평균에 대해 역전파...
        loss.mean().backward()
        self.optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    # 에피소드는 만 번...
    for n_epi in range(10000):
        done = False
        s, _ = env.reset()
        # 에피소드 별로 끝까지 돌기를 반복하는데...
        while not done:
            # 에피소드가 끝나지 않아도 10번 돌면 평가...
            for t in range(n_rollout):
                # 행동 확률...
                prob = model.pi(torch.from_numpy(s).float())
                # 정규화해서 확률로 만든다...이미 돼 있는거 아닌가? 이게 왜 필요할까? 아래 sample 함수를 쓰려고?
                m = Categorical(prob)
                # 확률로 행동 선택하고
                a = m.sample().item()
                # 다음 상태, 보상, 종료 여부 등 받고
                s_prime, r, done, truncated, info = env.step(a)
                # 경험치로 저장
                model.put_data((s, a, r, s_prime, done))

                # 다음 상태를 현재 상태로 하고 반복
                s = s_prime
                score += r

                if done:
                    break

            # 10개 경험 모이면 학습...
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score / print_interval
                )
            )
            # 스코어는 각 에피소드별로가 아니라 20 에피소드마다 초기화한다?
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
