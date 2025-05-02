from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple, MultiDiscrete
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.define_ids_to_discrete()

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        # 'map_info': MultiDiscrete(np.full(11*21, 117)),
        "bombs": Box(low=0, high=99, shape=(1,), dtype=np.int32),
        "seen_bombs": Box(low=0, high=99, shape=(1,), dtype=np.int32),
    })


    def reward_function(self, gamestate, last_gamestate, action):
        reward_val = 0
        # if gamestate["player_info"]["health"] <= 0:
        #     reward_val -= 5

        # if gamestate["screen_info"]["win"]:
        #     reward_val += 5

        if gamestate["player_info"]["bombs"] >= 2:
            reward_val += (last_gamestate["player_info"]["bombs"] - gamestate["player_info"]["bombs"])*1
        else :
            reward_val += (gamestate["player_info"]["bombs"] - last_gamestate["player_info"]["bombs"])*1

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

        print(observation)
        return observation

    
    def define_ids_to_discrete(self):
        type_vals = [0] + [163, 167] \
                    + list(range(194, 628)) \
                    + list(range(844, 896)) \
                    + list(range(899, 907))
        type_vals = sorted(set(type_vals))
        type_id_map = {}
        for i, v in enumerate(type_vals):
            type_id_map[v] = i
        self.type_id_map = type_id_map

        map_vals = [0] + list(range(1, 115)) + [190, 191]
        map_vals = sorted(set(map_vals))
        map_info_map = {}
        for i, v in enumerate(map_vals):
            map_info_map[v] = i
        self.map_info_map = map_info_map

        back_vals = [0, 564, 565, 567, 568, 570, 572, 574]
        back_vals = sorted(set(back_vals))
        back_map = {}
        for i, v in enumerate(back_vals):
            back_map[v] = i
        self.back_map = back_map