import numpy as np
import gymnasium as gym
import cv2 as cv
from gymnasium.spaces import Dict, Box, Discrete

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Box(
        low=0,
        high=255,
        shape=(256, 256, 3),  # (height, width, channels)
        dtype=np.uint8
    )


    action_space: gym.spaces.MultiDiscrete = gym.spaces.MultiDiscrete([
        3, # Movement X
        3, # Movement Y
        2, # Jump
    ])


    reset_options = {
        "ent_types_to_destroy": [600,601] + list(range(219, 342)) + list(range(899, 906))
    }

    data_to_send = []


    def action_to_input(self, action):
        return action + [0,0,0,1,0]

    def reward_function(self, gamestate, last_gamestate, action, info):
        truncated = False
        done = False

        # Too much time
        if gamestate["basic_info"]["time"] >= 60*30: # 30 seconds
            truncated = True

        reward_val = (last_gamestate["basic_info"]["money"] - gamestate["basic_info"]["money"]) / 1000.0
        return float(reward_val), done or truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        image = self.render()
        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
        return image
        
