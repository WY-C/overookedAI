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


class get_order_agent(Agent):
    def __init__(self, original_env, my_env, agent_idx=1):
        super().__init__()
        self.agent_idx = agent_idx
        self.original_env = original_env
        self.my_env = my_env
        self.O = ((0, 1), (4, 1))
        self.my_position = self.original_env.state.player_positions[0]
        self.bool_get_order = True
        self.order = ""
        self.target = ""
    def get_order(self):
        self.bool_get_order = False
        #이 order를 LLM 호출로 바꾸면 완료.
        order = input("what to do")
        #print(order)
        if(order == 'O'):
            #my_position = self.original_env.state.player_positions[0]
            distances = [(manhattan_distance(self.my_position, pos)) for pos in self.O]
            if distances[0] <= distances[1]:
                return order, 0
            else:
                return order, 1
        elif (order == 'P'):
            return order, 0
        elif (order == 'Stay'):
            return order, 0
        elif (order == 'D'):
            return order, 0
        else:
            return 'S', 0
            

    def action(self):
        self.my_position = self.original_env.state.player_positions[self.agent_idx]
        if self.bool_get_order:
            #print(1)
            self.order, self.target = self.get_order()
        #print(self.my_position)
        
        #둘이 다른경우: agent가 한칸 아래있음
        #print(self.my_position)
        #print(self.original_env.state.player_orientations[0])
        print(self.target)
        
        if(self.order == 'O'):
            #위로 한 칸 가야함
            if(self.my_position[1] != 1):
                action = 0
            elif (self.target == 0):
            #위치에 왔고, 방향도 맞음.
                if(self.my_position == (1, 1) and self.original_env.state.player_orientations[self.agent_idx] == (-1, 0)):
                    self.bool_get_order = True
                    action = 5
                else:
                    action = 3
            else:
                if(self.my_position == (3, 1) and self.original_env.state.player_orientations[self.agent_idx] == (1, 0)):
                    self.bool_get_order = True
                    action = 5
                else:
                    action = 2
        
        if(self.order == 'P'):
            #상하
            if(self.my_position[1] != 1):
                action = 0
            #상호작용
            elif (self.my_position == (2, 1) and self.original_env.state.player_orientations[self.agent_idx] == (0, -1)):
                self.bool_get_order = True
                action = 5
            elif (self.my_position == (2, 1)):
                action = 0
            else:
                if (self.my_position[0] < 2):
                    action = 2
                else:
                    action = 3
        if(self.order == 'Stay'):
            self.bool_get_order = True
            action = 4

        if(self.order == 'D'):
            if(self.my_position[1] != 2):
                action = 1
            #상호작용
            elif (self.my_position == (1, 2) and self.original_env.state.player_orientations[self.agent_idx] == (0, 1)):
                self.bool_get_order = True
                action = 5
            elif (self.my_position == (1, 2)):
                action = 1
            else:
                action = 3          
        if(self.order == 'S'):
            if(self.my_position[1] != 2):
                action = 1
            #상호작용
            elif (self.my_position == (3, 2) and self.original_env.state.player_orientations[self.agent_idx] == (0, 1)):
                self.bool_get_order = True
                action = 5
            elif (self.my_position == (3, 2)):
                action = 1
            else:
                action = 2              

        action_probs = self.a_probs_from_action(ACTION_MAP[int(action)])
        action = ACTION_MAP[int(action)]
        #print(action, action_probs)

        return action, {"action_probs": action_probs}  # Return as a list of actions
    
    def actions(self, states):
        return super().actions(states, self.agent_idx)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])