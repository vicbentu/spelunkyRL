from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple, MultiDiscrete
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.define_ids_to_discrete()

    ################## ENV CHARACTERISTICS ##################

    # 'entities': Tuple(tuple(
    #     Tuple((
    #         Tuple((
    #             Discrete(916),
    #             Discrete(3),
    #             Discrete(8),
    #             Discrete(916),
    #         )),
    #         Box(
    #             low=np.array([-17., -9.0, -10.0, -10.0]),
    #             high=np.array([17., 9.0, 10.0, 10.0]),
    #             shape=(4,),
    #             dtype=np.float64,
    #         ),
    #     ))
    #     for _ in range(150)))    
    # 

    observation_space = Dict({
        'health': Box(low=0, high=99, shape=(1,), dtype=np.int32),
        'powerups': MultiBinary(18),
        'bombs': Box(low=0, high=99, shape=(1,), dtype=np.int32),
        'ropes': Box(low=0, high=99, shape=(1,), dtype=np.int32),
        'money': Box(low=0, high=2e7, shape=(1,), dtype=np.int64),
        'holding': Discrete(446),
        'back': Discrete(8),

        'vel_x': Box(low=-10, high=10, shape=(1,), dtype=np.float64),
        'vel_y': Box(low=-10, high=10, shape=(1,), dtype=np.float64),
        'face_left': Discrete(2),
        'x_rest': Box(low=0, high=1, shape=(1,), dtype=np.float64),
        'y_rest': Box(low=0, high=1, shape=(1,), dtype=np.float64),

        'theme': Discrete(18),
        'level': Discrete(4),
        'time': Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
        'map_info': MultiDiscrete(np.full(11*21, 117)),
        'water_map': MultiDiscrete(np.full(11*21, 2)),
        'lava_map': MultiDiscrete(np.full(11*21, 2)),

        'entities': Tuple([Dict({
            'dx': Box(low=-10, high=10, shape=(1,), dtype=np.float64),
            'dy': Box(low=-5, high=5, shape=(1,), dtype=np.float64),
            'vx': Box(low=-10, high=10, shape=(1,), dtype=np.float64),
            'vy': Box(low=-10, high=10, shape=(1,), dtype=np.float64),
            'type_id': Discrete(507),
            'flip': Discrete(3),
            'holding': Discrete(446)
        }) for _ in range(75)]),
    })


    def reward_function(self, observation, last_observation):
        reward_val = 0
        if observation["player_info"]["health"] <= 0:
            reward_val -= 20
        else:
            hp_loss = last_observation["player_info"]["health"] - observation["player_info"]["health"]
            reward_val -= hp_loss * 1

        reward_val += (observation["player_info"]["bombs"] - last_observation["player_info"]["bombs"])*0.2
        reward_val += (observation["player_info"]["ropes"] - last_observation["player_info"]["ropes"])*0.2

        if observation["screen_info"]["win"]:
            reward_val += 20

        return float(reward_val)


    def gamestate_to_observation(self, gamestate):

        observation = {}
        # --- Player info ---
        player = gamestate["player_info"]
        observation["health"]   = np.array([player["health"]], dtype=np.int32)
        observation["powerups"] = np.array(player["powerups"], dtype=np.int32)  # MultiBinary(18)

        observation["bombs"] = np.array([player["bombs"]], dtype=np.int32)
        observation["ropes"] = np.array([player["ropes"]], dtype=np.int32)
        observation["money"] = np.array([player["money"]], dtype=np.int64)

        raw_holding = player.get("holding_type_player", 0)
        observation["holding"] = np.array([self.holding_map.get(raw_holding, 0)], dtype=np.int32)

        raw_back = player.get("back_item", 0)
        observation["back"] = np.array([self.back_map.get(raw_back, 0)], dtype=np.int32)

        observation["vel_x"] = np.array([player["vel_x"]], dtype=np.float64)
        observation["vel_y"] = np.array([player["vel_y"]], dtype=np.float64)

        # face_left => boolean => 0 or 1
        observation["face_left"] = np.array([1 if player["face_left"] else 0], dtype=np.int32)

        observation["x_rest"] = np.array([player["x_rest"]], dtype=np.float64)
        observation["y_rest"] = np.array([player["y_rest"]], dtype=np.float64)

        # --- Screen info ---
        screen = gamestate["screen_info"]

        observation["theme"] = np.array([screen["theme"]], dtype=np.int32)
        observation["level"] = np.array([screen["level"] - 1], dtype=np.int32)  
        observation["time"]  = np.array([screen["time"]], dtype=np.int32)

        # Initialize map_info
        map_info_arr  = np.zeros(11*21, dtype=np.int32)
        water_map_arr = np.zeros(11*21, dtype=np.int32)
        lava_map_arr  = np.zeros(11*21, dtype=np.int32)

        raw_map = screen["map_info"]   # shape = (11, 21, 3) presumably
        try:
            for i in range(11):
                for j in range(21):
                    idx = i * 21 + j
                    tile_id    = raw_map[i][j][0]
                    water_flag = raw_map[i][j][1]
                    lava_flag  = raw_map[i][j][2]
                    map_info_arr[idx]  = self.map_info_map.get(tile_id, 0)
                    water_map_arr[idx] = water_flag
                    lava_map_arr[idx]  = lava_flag
        except Exception as e:
            # create a dummy map if the map is not available
            for i in range(11):
                for j in range(21):
                    idx = i * 21 + j
                    map_info_arr[idx]  = 0
                    water_map_arr[idx] = 0
                    lava_map_arr[idx]  = 0      

        observation["map_info"]  = map_info_arr
        observation["water_map"] = water_map_arr
        observation["lava_map"]  = lava_map_arr

        # --- Entities ---
        # We have up to 75 entities, each = [dx, dy, vx, vy, type_id, flip, holding]
        # We'll create a tuple/list of dicts matching your 'entities' space definition.
        entity_list = gamestate["entity_info"]
        max_entities = 75

        processed_entities = []
        for i in range(max_entities):
            if i < len(entity_list):
                raw_ent = entity_list[i]
                # raw_ent = [dx, dy, vx, vy, type_id, flip, holding]
                dx, dy, vx, vy, raw_type, raw_flip, raw_hold = raw_ent

                ent_dict = {
                    "dx":     np.array([dx], dtype=np.float64),
                    "dy":     np.array([dy], dtype=np.float64),
                    "vx":     np.array([vx], dtype=np.float64),
                    "vy":     np.array([vy], dtype=np.float64),
                    "type_id": np.array([self.type_id_map.get(raw_type, 0)+1], dtype=np.int32),
                    "flip":    np.array([raw_flip+1], dtype=np.int32),
                    "holding": np.array([self.holding_map.get(raw_hold, 0)], dtype=np.int32)
                }
            else:
                # No entity => zero out
                ent_dict = {
                    "dx":     np.array([0], dtype=np.float64),
                    "dy":     np.array([0], dtype=np.float64),
                    "vx":     np.array([0], dtype=np.float64),
                    "vy":     np.array([0], dtype=np.float64),
                    "type_id": np.array([0], dtype=np.int32),
                    "flip":    np.array([0], dtype=np.int32),
                    "holding": np.array([0], dtype=np.int32)
                }
            processed_entities.append(ent_dict)

        observation["entities"] = tuple(processed_entities)

        return observation
    
    def define_ids_to_discrete(self):
        holding_vals = [0] + [163, 167] + list(range(194, 628)) + list(range(899, 907))
        holding_vals = sorted(set(holding_vals))  # ensure unique & sorted
        holding_map = {}
        for i, v in enumerate(holding_vals):
            holding_map[v] = i
        self.holding_map = holding_map

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


if __name__ == '__main__':
    from datetime import datetime
    env = SpelunkyEnv(speedup=True, frames_per_step=4)
    env.reset()

    startTime = datetime.now()

    counter = 0
    while True:
        counter += 1
        if counter % 60 == 0:
            print(f"{counter/60} seconds have passed, FPS: {counter/(datetime.now()-startTime).total_seconds()}")
        observation, reward, done, _, _ = env.step(env.action_space.sample())
        if done:
            print("Game Over")
            env.reset()