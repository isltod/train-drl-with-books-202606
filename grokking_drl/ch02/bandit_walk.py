P = {
    # 가장 바깥쪽 키가 상태 - 밴딧통로는 세칸
    0: {
        # 다음 키는 행동, 왼쪽 또는 오른쪽
        0: [
            # 그 안에 튜플은 가능한 전이 결과...
            (
                # 전이확률, 다음상태, 보상, 종료여부
                # 100% 확률로, 가운데 칸으로 가고, 보상은 없고, 종료되지 않음
                1.0,
                0,
                0.0,
                True,
            )
        ],
        1: [(1.0, 0, 0.0, True)],
    },
    1: {0: [(1.0, 0, 0.0, True)], 1: [(1.0, 2, 1.0, True)]},
    2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
}

print(P)

# 또는 gymnasium의 BanditWalk env를 사용할 수는 있는데...
import gymnasium as gym

# 뜬금없이pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk 설치하고 import 해주고...
# noinspection PyUnusedImports
import gym_walk

# 환경은 unwrapped로 풀어서 받아놓고
env = gym.make("BanditWalk-v0").unwrapped
# 거기서 P를 호출하는데, 줄이가고...
P = env.P
# 나온 결과도 비슷하지만 꽤 다른 모양으로 출력된다...
print(P)
