from my_env.Rllib_multi_agent import Rllib_multi_agent

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.ppo import PPO
import os
ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}


my_env = Rllib_multi_agent()


from ray.tune.registry import register_env

def env_creator(config):
    return Rllib_multi_agent(config)

register_env("Rllib_multi_agent", env_creator)




ray.init()

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return "shared_policy"

sample_obs = my_env._get_obs(0)
flattened_shape = sample_obs['agent_1'].flatten().shape

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

def save_checkpoint_with_reward(trainer, reward, results, base_dir="./checkpoints"):
    # reward를 소수점 둘째 자리까지 문자열로 변환, 소수점 대신 언더바로 변경(폴더명 안전하게)
    reward_str = f"{reward:.2f}".replace('.', '_')

    # 저장할 폴더 경로 생성
    save_dir = os.path.join(base_dir, f"reward_{reward_str}_rewardshaping")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 체크포인트 저장
    checkpoint_path = trainer.save(checkpoint_dir=save_dir)
    #print(f"Checkpoint saved at: {checkpoint_path}")
      # --- 원하는 핵심 정보만 선택하여 출력 ---
    print("\n--- 🎉 New Best Model Saved! ---")
    print(f"  Iteration:     {results['training_iteration']}")
    print(f"  Mean Reward:   {reward:.2f}")
    print(f"  Policy Loss:   {results['learners']['shared_policy']['policy_loss']:.4f}")
    print(f"  Value Loss:    {results['learners']['shared_policy']['vf_loss']:.4f}")
    #print(f"  Saved to:      {checkpoint_path}")
    print("---------------------------------\n")
    return checkpoint_path


        

rllib_trainer = PPO(config=config)
best_mean_reward = -1


while True:
    results = rllib_trainer.train()
    iteration = results["training_iteration"]
    current_mean_reward = results["env_runners"]["episode_return_mean"]

    if current_mean_reward > best_mean_reward:
        best_mean_reward = current_mean_reward
        
        # 모델을 저장하고, 저장 결과 객체를 받습니다.
        save_checkpoint_with_reward(rllib_trainer, current_mean_reward, results)
        # 저장 결과에서 실제 파일 경로만 추출합니다.
       


    else:
        # 최고 기록을 갱신하지 못한 경우, 현재 상태만 간단히 출력
        print(
            f"Iter: {results['training_iteration']:<3} | "
            f"Reward: {current_mean_reward:<7.2f} | "
            f"Best: {best_mean_reward:<7.2f} | "
            f"Timesteps: {results['num_env_steps_sampled_lifetime']}"
        )
