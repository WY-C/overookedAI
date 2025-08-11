from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from stable_baselines3 import PPO, DQN
import numpy as np

ACTION_MAP = {
    0: (0, -1),   # NORTH
    1: (0, 1),    # SOUTH
    2: (1, 0),    # EAST
    3: (-1, 0),   # WEST
    4: (0, 0),    # STAY
    5: "interact" # INTERACT
}


class SB3Agent(Agent):
    def __init__(self, env, model_path = None, agent_idx=0, mode = "PPO"):
        super().__init__()
        self.model_path = model_path
        self.agent_idx = agent_idx
        self.model = PPO("MlpPolicy", env, verbose=1, device="cpu")#learning_rate=0.001, n_steps=2048, batch_size = 64, ent_coef =0.1, gamma=0.99, max_grad_norm=0.1)
        DQN_model = DQN("MlpPolicy", env, verbose=1, device="cpu")#, learning_rate=0.001, batch_size = 6, gamma=0.99, max_grad_norm=0.1)
        self.mode = mode
        if mode == "DQN":
            self.model = DQN_model
        self.env = env

    def load_model(self):
        if self.model_path:
            if self.mode == "PPO":
                self.model = PPO.load(self.model_path, env=self.env, device="cpu")
                self.model.set_env(self.env)
            elif self.mode == "DQN":
                self.model = DQN.load(self.model_path, env=self.env, device="cpu")
                self.model.set_env(self.env)
        else:
            raise ValueError("Model path is not provided.")
        
    def save_model(self, save_path):
        if self.model:
            self.model.save(save_path)
        else:
            raise ValueError("Model is not trained or loaded.")
    def action(self, state):
        if isinstance(state, OvercookedState):
            # Convert OvercookedState to the format expected by the model
            obs = self.env._get_obs()
        # Convert the state to the format expected by the model
        if isinstance(obs, tuple):
        # 튜플을 하나의 배열로 합치거나 첫 번째 요소만 사용
            obs = np.array(obs).flatten()
        action = self.model.predict(obs)[0]
        action_probs = self.a_probs_from_action(ACTION_MAP[int(action)])
        actions = ACTION_MAP[int(action)]
        #print(self.overcooked_env.display_states(state))
        #print(1)
        #print(action, action_probs)

        return actions, {"action_probs": action_probs}  # Return as a list of actions
    
    def actions(self, states):
        return super().actions(states, self.agent_idx)

    def learn(self, total_timesteps, overcooked_env, state, callback):
        #print(overcooked_env.display_states(state))
        #print(state.player_positions)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, state):
        if isinstance(state, OvercookedState):
            # Convert OvercookedState to the format expected by the model
            obs = self.env._get_obs()
        # Convert the state to the format expected by the model
        if isinstance(obs, tuple):
        # 튜플을 하나의 배열로 합치거나 첫 번째 요소만 사용
            obs = np.array(obs).flatten()
        return self.model.predict(obs)[0]

# class myAgent():
#     def __init__(self, map):
#         self.llm_API
#         self.map = map
#         self.location = (1, 3)
#         #O, D, P, S의 위치 모두 기록

#     def get_action(self, location):
#         self.location = location
#         self.result = "O"
#         return self.result
    
#     def action(self, map, location):
#         if(self.result == "O"): 




#     def process_end(self):
    