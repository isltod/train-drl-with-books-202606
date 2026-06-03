import random

import numpy as np


class GridWorld:
    # 앞의 GridWorld와 구조는 같고, 중간에 벽 블록이 6개 있는 것만 틀린 클래스
    def __init__(self):
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

    def move_left(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1

    def move_right(self):
        if self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        elif self.y == 6:
            pass
        else:
            self.y += 1

    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1

    def is_done(self):
        if self.x == 4 and self.y == 6:  # 목표 지점인 (4,6)에 도달하면 끝난다
            return True
        else:
            return False

    def reset(self):
        self.x = 0
        self.y = 0
        return self.x, self.y


class QAgent:
    # 여기서는 v 대신 q를 이용하고, 가치만 계산하는 것이 아니라 최적 정책을 학습하므로...
    # q를 여기에 두고, 정책을 학습한다...
    def __init__(self):
        # q 가치를 저장하는 테이블, (세로, 가로, 행동 종류)...
        self.q_table = np.zeros((5, 7, 4))
        # 엡실론 그리디 정책
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self, s):
        # 현재 위치를 상태로 받고,
        x, y = s
        coin = random.random()
        # eps-greedy 확률로 액션을 선택
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            # 그리디 옵션이면 현재 위치에서 최대 가치를 가진 행동 방향을 무조건 선택
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아...
        cum_reward = 0
        # 종착지로부터 역순으로 돌면서
        for transition in history[::-1]:
            # 현재 위치, 행동, 보상, 나중 위치
            s, a, r, s_prime = transition
            x, y = s
            # 종료지점부터 현재까지의 누적 보상(γ=1이니까)으로 q 가치의 기댓값을 α 비율로 업데이트
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (
                cum_reward - self.q_table[x, y, a]
            )
            # 다음 계산을 위해 현재 보상을 누적 추가
            cum_reward = cum_reward + r

    def anneal_eps(self):
        # 처음에 90%로 시작해서 3%씩 탐색비율 줄여가기
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)


def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000):  # 총 1,000 에피소드 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done:  # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history)  # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps()

    agent.show_table()  # 학습이 끝난 결과를 출력


if __name__ == "__main__":
    main()
