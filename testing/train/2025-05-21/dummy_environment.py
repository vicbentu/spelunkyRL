import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        "can_jump"  : Discrete(2)
    })

    action_space: gym.spaces.MultiDiscrete = gym.spaces.MultiDiscrete([
        3, # Movement X
        3, # Movement Y
        2, # Jump
    ])

    reset_options = {
        "ent_types_to_destroy": [600,601] + list(range(219, 342)) + list(range(899, 906)),
        "manual_control": False,
        "god_mode": True,
    }

    additional_data = [
        "map_info",
        "dist_to_goal",
        "entity_info"
    ]

    def action_to_input(self, action):
        return action + [0,0,0,1,0]

    def reward_function(self, gamestate, last_gamestate, action, info):
        truncated = False
        done = False
        reward_val = -0.01  # small penalty for each step to encourage faster completion

        if gamestate["basic_info"]["time"] >= 60*90: # 900 steps
            truncated = True
            reward_val -= 5

        return float(reward_val), done and not truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        observation = {}
        observation["can_jump"] = np.int32(1)

        return observation