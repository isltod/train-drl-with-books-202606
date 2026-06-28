P = {
    0: {0: [(1.0, 0, 0.0, True)], 1: [(1.0, 0, 0.0, True)]},
    1: {
        # 기본형과 다르게 0.8과 0.2 확률로 갈리고, 다음 위치가 0, 2로 갈리고, 보상도 갈리고...어쨌든 종료된다..
        0: [(0.8, 0, 0.0, True), (0.2, 2, 1.0, True)],
        1: [(0.8, 2, 1.0, True), (0.2, 0, 0.0, True)],
    },
    2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
}

print(P)

# gymnasium 환경을 쓰는 방법은 비슷하고 이름만 BanditSlipperyWalk
import gymnasium as gym

# noinspection PyUnusedImports
import gym_walk

env = gym.make("BanditSlipperyWalk-v0").unwrapped
E = env.P
print(E)
