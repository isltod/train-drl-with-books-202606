import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 이 단순한 모델은 모두 완전연결인데, 첫 번째는 입력 층
        self.fc1 = nn.Linear(1, 128)
        # 두 개의 히든? 아니면 출력 층까지 3개의 히든? 그것도 아니면 출력층 빼고 입력부터 3개의 히든?
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        # 이건 따로 안하고 층에 activation으로 넣지 않나?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def true_fun(X):
    # U(0,1)에 0.4 곱하고 -0.2하면 U(-0.2,0.2)
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise


def plot_results(model):
    x = np.linspace(0, 5, 100)
    # 결과를 바로 받으려고 입력을 토치 텐서로 바꿔놓고, 모델에 넣을 때는 배치 차원 추가...
    input_x = torch.from_numpy(x).float().unsqueeze(1)
    plt.plot(x, true_fun(x), label="Truth")
    # 여기서 그릴 때 바로 모델에서 받는다...
    plt.plot(x, model(input_x).detach().numpy(), label="Prediction")
    plt.legend(loc="lower right", fontsize=15)
    plt.xlim((0, 5))
    plt.ylim((-1, 5))
    plt.grid()
    plt.show()


def main():
    data_x = np.random.rand(10000) * 5  # 0~5 사이 숫자 1만개를 샘플링하여 인풋으로 사용
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for step in tqdm(range(10000)):
        batch_x = np.random.choice(
            data_x, 32
        )  # 랜덤하게 뽑힌 32개의 데이터로 mini-batch를 구성, 배치 차원 추가...
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
        pred = model(batch_x_tensor)

        batch_y = true_fun(batch_x)
        # 비교할 정답지도 배치 차원 달린 토치 텐서로 넣어주기...
        truth = torch.from_numpy(batch_y).float().unsqueeze(1)
        loss = F.mse_loss(pred, truth)  # 손실 함수인 MSE를 계산하는 부분

        optimizer.zero_grad()
        # 역전파...배치니까 그냥 loss가 아니라 그 평균으로 한다...
        loss.mean().backward()
        optimizer.step()  # 실제로 파라미터를 업데이트 하는 부분

    plot_results(model)


if __name__ == "__main__":
    main()
