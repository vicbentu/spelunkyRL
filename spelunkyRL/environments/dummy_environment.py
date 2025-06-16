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
        "ent_types_to_destroy": [],
        "manual_control": True,
        "god_mode": True,

    }

    data_to_send = []

    def action_to_input(self, action):
        return action + [0,0,0,1,0]

    def reward_function(self, gamestate, last_gamestate, action, info):
        truncated = False
        if gamestate["basic_info"]["time"] >= 60*90: # 900 steps
            truncated = True

        return float(0), truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        observation = {}
        observation["can_jump"] = np.int32(1)

        return observation