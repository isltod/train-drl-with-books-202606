from continuous_observ_space import *
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
from tqdm import tqdm


# 타일 코딩은 경계를 겹치게 잡는 방법이라는데...
class TileCodingEnv(gym.ObservationWrapper):
    def __init__(self, env, bins, low, high, n=4):
        super().__init__(env)
        self.n = n  # 타일링 개수
        self.bins = bins
        self.low = low
        self.high = high
        # state aggregation은 여기서 그냥 Discrete 만들어서 bucket으로 사용했는데...
        # tile coding에서 상태 공간은 (n, bins[0], bins[1]) 형태가 된다고...이게 조금씩 앞 뒤와 겹치는 모양...
        # Gym Space로는 표현하기 복잡해서 그렇다고 정해두기만 하고 observation에서 튜플 리스트를 반환하도록 함

        # 각 타일링별 오프셋(Offset) 생성
        # 각 차원의 구간 너비 계산
        self.tile_width = (high - low) / bins
        self.offsets = []
        for i in range(n):
            # 타일링마다 조금씩 위치를 이동시킴
            offset = low + i * (self.tile_width / n)
            self.offsets.append(offset)

        # 각 타일링별 버킷 경계 생성
        self.tilings_buckets = []
        for i in range(n):
            # 오프셋만큼 이동된 경계선 생성
            # 주의: 상태가 범위를 벗어날 수 있으므로 넉넉하게 처리하거나 클리핑 필요
            # 여기서는 간단하게 linspace로 생성하되, 입력값 처리시 digitize가 알아서 처리함
            buckets = [
                np.linspace(l, h, b - 1)
                for l, h, b in zip(
                    self.offsets[i], self.offsets[i] + (high - low), bins
                )
            ]
            self.tilings_buckets.append(buckets)

    def observation(self, obs):
        # n개의 타일링 각각에 대해 인덱스를 계산
        indices = []
        for i in range(self.n):
            # i번째 타일링의 버킷을 이용해 인덱스 추출
            # 범위를 벗어나는 경우를 대비해 bins-1로 클리핑해주는 것이 안전함
            idx = tuple(
                np.clip(np.digitize(x, b), 0, bn - 1)
                for x, b, bn in zip(obs, self.tilings_buckets[i], self.bins)
            )
            indices.append(idx)
        return tuple(indices)


# 래퍼 적용 (4개의 타일링 사용)
tilings = 4
bins = np.array([20, 20])
tcenv = TileCodingEnv(env, bins=bins, low=low, high=high, n=tilings)

# 샘플 상태 확인
state, _ = tcenv.reset()
print(f"Sample state (Tile Coding): {state}")
# 결과 예: ((idx1_x, idx1_v), (idx2_x, idx2_v), ...) 형태로 4개의 좌표가 나옴
