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
        'map_info': Box(low=0, high=116, shape=(11, 21), dtype=np.int32),
    })


    def reward_function(self, gamestate, last_gamestate, action):
        reward_val = 0
        if gamestate["player_info"]["health"] <= 0:
            reward_val -= 5

        if gamestate["screen_info"]["win"]:
            reward_val += 5

        reward_val += (last_gamestate["screen_info"]["dist_to_goal"] - gamestate["screen_info"]["dist_to_goal"])*0.1

        return float(reward_val)

    def gamestate_to_observation(self, gamestate):

        observation = {}
        # --- Player info ---
        # observation["bombs"] = np.array([gamestate["player_info"]["bombs"]], dtype=np.float32)
        # --- Screen info ---
        screen = gamestate["screen_info"]

        # Initialize map_info
        screen_map = np.array(screen["map_info"])      # (11, 21, 3)
        observation["map_info"] = screen_map[:, :, 0]  # (11, 21)

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

        # # --- Entities ---
        # observation["seen_bombs"] = np.array([0], dtype=np.float32)
        # entity_list = gamestate["entity_info"]  # a list of up to N=75 raw entity data
        # for i in range(min(75, len(entity_list))):
        #     dx, dy, vx, vy, raw_type, raw_flip, raw_hold = entity_list[i]

        #     # observation["seen_bombs"] ++
        #     observation["seen_bombs"] += 1 if raw_type == 347 else 0

        # print(observation)
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