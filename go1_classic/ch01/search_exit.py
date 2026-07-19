from envs import Maze
from matplotlib import pyplot as plt
from pprint import pprint
import numpy as np

# 환경 재생성
env = Maze()
initial_state, _ = env.reset()

print(f"예시 - 초기 상태: {initial_state}")
print(f"상태 공간(State Space) 타입: {env.observation_space}")
# MultiDiscrete([5 5])는 첫 번째 차원 5개, 두 번째 차원 5개의 값을 가짐을 의미

# spaces.Discrete에서 sample()하면 그 중 하나를 뽑아준다...
print(f"유효한 행동 예시 (샘플링): {env.action_space.sample()}")
print(f"행동 공간(Action Space) 타입: {env.action_space}")

# trajectory 만들기
state, _ = env.reset()
trajectory = []
for _ in range(3):
    # 1. 행동 선택
    action = env.action_space.sample()
    # 2. 스텝
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    # 3. 경험 저장
    trajectory.append([state, action, reward, done, next_state])
    # 4. 현재를 다음 상태로...반복
    state = next_state
env.close()
pprint(f"생성된 첫 번째 궤적:\n{trajectory}")

# 에피소드 - 종료 상태까지 궤적
state, _ = env.reset()
episode = []
done = False
step_count = 0
# 행동이 랜덤 선택이므로 까딱하면 무한루프처럼 될 수 있어서, 안전장치로 최대 1000스텝 제한
while not done and step_count < 1000:
    # 나머지는 위 절차와 동일 - 행동, 스텝, 저장, 다음 상태로 반복
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode.append([state, action, reward, done, next_state])
    state = next_state
    step_count += 1
env.close()
print(f"생성된 에피소드 길이: {len(episode)} 스텝")

# 즉시보상 reward
state, _ = env.reset()
action = env.action_space.sample()
_, reward, _, _, _ = env.step(action)
print(f"상태 {state}에서 행동 {action}을 취해 즉시보상 {reward}를 얻음")

# 총보상 return - 즉시보상에 할인율 γ 적용
state, _ = env.reset()
done = False
gamma = 0.99
G_t = 0
t = 0
max_steps = 2000  # 무한 루프 방지

while not done and t < max_steps:
    action = env.action_space.sample()
    _, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    # 할인율을 적용하여 총보상을 누적하는 방법...
    G_t += (gamma**t) * reward
    t += 1
env.close()
print(f"""출구를 찾는데 {t}번의 이동이 걸렸으며, 
각 보상은 -1이므로, 총보상은 {G_t}이다.""")


# uniform dist 랜덤 정책 - 원래 정책 π는 state를 보고 결정해야 하는데, 여긴 그냥 uniform dist 예시...
def random_policy(state):
    return np.array([0.25] * 4)


# 행동 정책을 그래프로...
action_probabilities = random_policy(state)
objects = ("Up", "Right", "Down", "Left")
x_ticks = np.arange(len(objects))
plt.bar(x_ticks, action_probabilities, alpha=0.5)
plt.xticks(x_ticks, objects)
plt.ylabel("π(a|s)")
plt.title("Random Policy Probability")
plt.show()

# 정책을 이용한 에피소드 시각화...
state, _ = env.reset()
done = False

# 시각화를 위한 초기 이미지 설정
img = plt.imshow(env.render())
plt.axis("off")
# 연속 그래프 모드 켜기
plt.ion()

step = 0
# 너무 오래 걸릴 수 있으므로 50스텝만 시각화하거나 종료될 때까지 실행
while not done and step < 50:
    # 1. 정책에 따라 행동 선택 (확률 분포 기반 랜덤 선택)
    action = np.random.choice(range(4), 1, p=action_probabilities)[0]

    # 2. 환경에 행동 적용
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # 3. 화면 갱신
    img.set_data(env.render())
    # 요게 연속으로 그리는데...
    plt.draw()
    plt.pause(0.1)
    step += 1

env.close()
# 연속 그래프 모드 끄기...plt.show는 그 뒤에?
plt.ioff()
# plt.show()

print("시뮬레이션 종료")
