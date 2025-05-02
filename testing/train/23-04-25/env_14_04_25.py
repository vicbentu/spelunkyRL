from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple, MultiDiscrete
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        # 'map_info': MultiDiscrete(np.full(11*21, 117)),
        "bombs": Box(low=0, high=99, shape=(1,), dtype=np.int32),
        "seen_bombs": Box(low=0, high=99, shape=(1,), dtype=np.int32),
    })


    def reward_function(self, gamestate, last_gamestate, action):
        reward_val = 0
        if gamestate["player_info"]["bombs"] >= 2:
            reward_val += (last_gamestate["player_info"]["bombs"] - gamestate["player_info"]["bombs"])*1
        else :
            reward_val += (gamestate["player_info"]["bombs"] - last_gamestate["player_info"]["bombs"])*1

        if action[0] == 1 and gamestate["player_info"]["bombs"] == last_gamestate["player_info"]["bombs"]:
            reward_val -= 1

        return float(reward_val)

    def gamestate_to_observation(self, gamestate):

        observation = {}
        # --- Player info ---
        # observation["bombs"] = gamestate["player_info"]["bombs"]
        observation["bombs"] = np.array([gamestate["player_info"]["bombs"]], dtype=np.float32)
        # --- Screen info ---
        # screen = gamestate["screen_info"]

        # # Initialize map_info
        # map_info_arr  = np.zeros(11*21, dtype=np.int32)
        # raw_map = screen["map_info"]   # shape = (11, 21, 3) presumably

        # try:
        #     for i in range(11):
        #         for j in range(21):
        #             idx = i * 21 + j
        #             tile_id    = raw_map[i][j][0]
        #             map_info_arr[idx]  = self.map_info_map.get(tile_id, 0)
        # except Exception as e:
        #     # create a dummy map if the map is not available
        #     for i in range(11):
        #         for j in range(21):
        #             idx = i * 21 + j
        #             map_info_arr[idx]  = 0

        # observation["map_info"]  = map_info_arr

        # --- Entities ---
        observation["seen_bombs"] = np.array([0], dtype=np.float32)
        entity_list = gamestate["entity_info"]  # a list of up to N=75 raw entity data
        for i in range(min(75, len(entity_list))):
            dx, dy, vx, vy, raw_type, raw_flip, raw_hold = entity_list[i]

            # observation["seen_bombs"] ++
            observation["seen_bombs"] += 1 if raw_type == 347 else 0

        return observation