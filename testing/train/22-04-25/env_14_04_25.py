from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple, MultiDiscrete
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        # 'map_info': MultiDiscrete(np.full(11*21, 117)),
        'map_info': Box(low=0, high=116, shape=(11, 21, 15), dtype=np.int32),
        # 'map_info': Box(low=0, high=116, shape=(11, 21), dtype=np.int32),
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
        screen = gamestate["screen_info"]
        observation["map_info"] = np.array(screen["pos_type_matrix"])  # (11, 21, 15)

        # with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
        #     f.write("TEST\n")
        #     f.write(f"map_info shape: {observation['map_info'].shape}\n")
            # f.write(f"map_info: {observation['map_info']}\n")
            # f.write("map_info:\n")
            # for i in range(11):
            #     for j in range(21):
            #         idx = i * 21 + j
            #         tile_id    = screen_map[i][j][0]
            #         f.write(f"{tile_id:3d} ")
            #     f.write("\n")

        return observation


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