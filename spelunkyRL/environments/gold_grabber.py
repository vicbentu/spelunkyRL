import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        'map_info': Box(low=0, high=116, shape=(1, 11, 21), dtype=np.int32),
        'gold_info': Box(low=0, high=np.inf, shape=(1, 11, 21), dtype=np.int32),
        "char_state": Discrete(23),
        "can_jump"  : Discrete(2),
    })

    action_space: gym.spaces.MultiDiscrete = gym.spaces.MultiDiscrete([
        3, # Movement X
        3, # Movement Y
        2, # Jump
    ])


    reset_options = {
        "ent_types_to_destroy": [600,601] + list(range(219, 342)) + list(range(899, 906))
    }

    data_to_send = [
        "map_info",
        "entity_info"
    ]


    def action_to_input(self, action):
        return action + [0,0,0,1,0]

    def reward_function(self, gamestate, last_gamestate, action, info):
        truncated = False
        done = False

        if gamestate["basic_info"]["time"] >= 60*30: # 30 seconds
            truncated = True

        reward_val = (last_gamestate["basic_info"]["money"] - gamestate["basic_info"]["money"]) / 1000.0
        return float(reward_val), done or truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        observation = {}

        observation["map_info"] = np.array(gamestate["map_info"])
        observation["char_state"] = np.int32(gamestate["basic_info"]["char_state"])
        observation["can_jump"] = np.int32(int(gamestate["basic_info"]["can_jump"]))


        observation["gold_info"] = np.zeros((1, 11, 21), dtype=np.int32)
        for ent in gamestate["entity_info"]:
            if ent[4] >= 495 and ent[4] <= 506:
                x = int(ent[0] + gamestate["basic_info"]["x_rest"] + 10.5)
                y = 11 - int(ent[1] + gamestate["basic_info"]["y_rest"] + 5.5)
                if x < 0 or x >= 21 or y < 0 or y >= 11:
                    print(f"Entity out of bounds: {ent}, x: {x}, y: {y}")
                    continue
                observation["gold_info"][0, y, x] += [
                    500, 1500, 5000, 800, 1200, 1600,
                    500, 500, 200, 300, 400, 100
                ][ent[4] - 495]

        return observation