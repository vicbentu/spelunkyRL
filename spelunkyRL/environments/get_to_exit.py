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
        "char_state": Discrete(23),
        "can_jump"  : Discrete(2)
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
        "dist_to_goal"
    ]


    def action_to_input(self, action):
        return action + [0,0,0,1,0]

    def reward_function(self, gamestate, last_gamestate, action, info):
        truncated = False
        done = False
        reward_val = -0.01  # small penalty for each step to encourage faster completion

        # Too much time
        if gamestate["basic_info"]["time"] >= 60*90: # 900 steps
            truncated = True
            # reward_val -= 5

        # Level completed
        if gamestate["dist_to_goal"] <= 1:
            done = True
            reward_val += (((60*90) - gamestate["basic_info"]["time"]) / (60*90))*5
            info["success"] = True
            info["time"] = gamestate["basic_info"]["time"]
        
        # No progress, clipping
        if gamestate["dist_to_goal"] < getattr(self, "min_dist_to_goal", float("inf")):
            self.min_dist_to_goal = gamestate["dist_to_goal"]
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
        if self.no_improve_counter >= 200:
            truncated = True
            reward_val -= 5

        # Penalize for the rest of steps
        if truncated:
            timesteps = gamestate["basic_info"]["time"] / 6
            reward_val -= 0.01 * (900 - timesteps)
        if done or truncated:
            self.min_dist_to_goal = float("inf")

        # Reward getting close to the goal
        reward_val += (last_gamestate["dist_to_goal"] - gamestate["dist_to_goal"])*0.1
        
        return float(reward_val), done or truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        observation = {}

        observation["map_info"] = np.array(gamestate["map_info"])
        observation["char_state"] = np.int32(gamestate["basic_info"]["char_state"])
        observation["can_jump"] = np.int32(int(gamestate["basic_info"]["can_jump"]))

        return observation