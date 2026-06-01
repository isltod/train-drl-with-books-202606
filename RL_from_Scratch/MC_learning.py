import random

from tqdm import tqdm


class GridWorld:
    def __init__(self):
        # 격자 x, y 좌표
        self.x = 0
        self.y = 0

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()

        reward = -1  # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1
        if self.y > 3:
            self.y = 3

    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False

    def get_state(self):
        return self.x, self.y

    def reset(self):
        self.x = 0
        self.y = 0
        return self.x, self.y


class Agent:
    def __init__(self):
        self.toss = random.random

    # 책 소스대로 하면 static 아니냐는 경고 나와서 self 변수 참조하도록 수정...
    def select_action(self):
        coin = self.toss()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    env = GridWorld()
    agent = Agent()
    data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    # 에피소드를 5만번 반복하는데...
    for k in tqdm(range(50000)):
        done = False
        history = []

        # 한 에피소드 끝날때까지 반복해서
        while not done:
            # 행동 선택하고, 상태/보상/종료 받고, 그걸 경험으로 저장
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        # 한 에피소드 끝나면 환경을 초기화
        env.reset()

        cum_reward = 0
        # 경험을 역순으로 재현해서...
        for transition in history[::-1]:
            x, y, reward = transition
            # 112쪽 식으로 격자의 상태 가치를 업데이트하고...
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            # 다음 격자로 상태 가치를 누적 전달 - G8 = R8 + γR9 -> G7 = R7 + γG8 = R7 + γR8 + γ^2R9
            cum_reward = reward + gamma * cum_reward

    for row in data:
        print(row)


if __name__ == "__main__":
    main()
