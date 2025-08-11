import ray
import time
from ray.rllib.algorithms.ppo import PPO
from my_env.Rllib_multi_agent import Rllib_multi_agent # 사용자의 환경 클래스를 import
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

import os

# 1. 불러올 체크포인트 경로
CHECKPOINT_PATH = os.path.abspath() # 본인의 경로로 수정하세요.
#/tmp/tmpayhhrz5h
#/tmp/tmpcpnress2 tmppso2qtor
# Ray 초기화
my_env = Rllib_multi_agent()
sample_obs = my_env._get_obs(0)
flattened_shape = sample_obs['agent_1'].flatten().shape

ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}
import gymnasium as gym
import numpy as np
ray.init()

def env_creator(config):
    return Rllib_multi_agent(config)

register_env("Rllib_multi_agent", env_creator)

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):

    return "shared_policy"
config = (
    PPOConfig()
    .environment(env="Rllib_multi_agent")
    .framework("torch")
    .env_runners(
        # 이 값을 환경의 horizon보다 작거나 비슷한 값으로 설정해보세요.
        # 'auto'가 기본값이지만, 명시적으로 설정하는 것이 디버깅에 도움이 될 수 있습니다.
        rollout_fragment_length=400
    )
    .multi_agent(
        policies={
            "shared_policy": (
                None,
                gym.spaces.Box(low=0, high=1, shape=flattened_shape, dtype=np.float32),
                gym.spaces.Discrete(len(ACTION_MAP)),
                {}
            )
        },
        policy_mapping_fn=policy_mapping_fn,
    )
)


# 2. 체크포인트로부터 학습된 트레이너(알고리즘)를 복원합니다.
#    PPO 대신 다른 알고리즘을 사용했다면 해당 알고리즘 클래스를 사용해야 합니다.
print(f"체크포인트 로딩 중... 경로: {CHECKPOINT_PATH}")
restored_trainer = PPO.from_checkpoint(CHECKPOINT_PATH)
print("로딩 완료!")

# 3. 평가를 위한 새로운 환경 인스턴스를 생성합니다.
env = Rllib_multi_agent()
obs, info = env.reset()

# 에피소드 초기화
terminated = {"__all__": False}
total_reward = 0

print("\n--- 학습된 모델 평가 시작 ---")

# 4. 한 에피소드가 끝날 때까지 반복합니다.
while not terminated["__all__"]:
    # 5. 현재 관측값(obs)을 사용해 학습된 모델로부터 행동을 계산합니다.
    #    정책 공유를 사용했으므로 policy_id='shared_policy'로 지정합니다.
    action_dict = restored_trainer.compute_actions(observations = obs)

    # 6. 환경에서 행동을 실행하고 다음 상태 정보를 받습니다.
    obs, rewards, terminated, truncated, infos = env.step(action_dict)
    
    # 7. 환경을 렌더링하여 현재 상태를 화면에 표시합니다.
    #    (render 메서드가 텍스트 기반이면 텍스트가, 시각화 기반이면 이미지가 나옵니다)
    print("\n" + "="*30)
    env.render()
    print(f"Action: {action_dict}, Reward: {rewards['agent_0']:.2f}")
    print("="*30)

    total_reward += rewards['agent_0']
    
    # 너무 빨리 지나가지 않도록 잠시 멈춤
    time.sleep(0.2)

    # 에피소드가 조기 종료(truncated)된 경우에도 루프를 빠져나옵니다.
    if truncated["__all__"]:
        break

print("\n--- 평가 종료 ---")
print(f"에피소드 총 보상: {total_reward}")

# 자원 정리
restored_trainer.stop()
ray.shutdown()