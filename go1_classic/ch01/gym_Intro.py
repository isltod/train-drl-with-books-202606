import matplotlib.pyplot as plt
from envs import Maze

if __name__ == "__main__":
    # 1. 환경 생성 - gym.make() 또는 이렇게 알아서 초기화...
    env = Maze()

    # 2. 환경 초기화, info는 추가 정보를 담은 딕셔너리
    initial_state, info = env.reset()
    print(f"새로운 에피소드가 시작되는 상태: {initial_state}")
    print(f"상태 추가 정보: {info}")

    # 3. 시각화 - 현재 상태를 ndarray 형태로 주는데, 이걸 이미지로 처리하면 그림이 되는 모양...
    frame = env.render()
    plt.axis("off")
    plt.title(f"State: {initial_state}")
    plt.imshow(frame)
    plt.show()

    # 4. 행동 수행 - step으로..., gymnasium 오면서 truncated 추가해 5가지 반환 값
    action = 2  # 2번 행동 (아래로 이동) 선택
    next_state, reward, terminated, truncated, info = env.step(action)

    # 에피소드 종료 여부는 terminated 혹은 truncated가 True일 때
    done = terminated or truncated

    print(f"아래로 한 칸 이동 후 상태: {next_state}")
    print(f"이동에 대한 보상: {reward}")
    print("태스크 완료 여부:", "완료됨" if done else "완료되지 않음")

    # 5. 새로운 상태 시각화
    frame = env.render()
    plt.axis("off")
    plt.title(f"State: {next_state}")
    plt.imshow(frame)
    plt.show()

    # 6. 환경 종료...근데 위에서 close 메서드는 안 만들었는데...gym.env에서 상속받은 모양...
    env.close()
