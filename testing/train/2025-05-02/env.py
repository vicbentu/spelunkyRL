from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple, MultiDiscrete
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        'map_info': Box(low=0, high=116, shape=(5, 11, 21), dtype=np.int32),
        "char_state": Discrete(23),
        "can_jump"  : Discrete(2)
        # "char_state": Box(low=0, high=22, shape=(1,), dtype=np.int32),
        # "can_jump"  : Box(low=0, high=1, shape=(1,), dtype=np.int32),

    })


    def reward_function(self, gamestate, last_gamestate, action, done):
        truncated = False
        reward_val = -0.01  # small penalty for each step to encourage faster completion

        # Clipping
        if gamestate["screen_info"]["time"] >= 60*90: # 900 steps
            truncated = True
            reward_val -= 5
        if gamestate["screen_info"]["dist_to_goal"] < 1:
            done = True
            reward_val += 5
        
        # No progress clipping
        if gamestate["screen_info"]["dist_to_goal"] < getattr(self, "min_dist_to_goal", float("inf")):
            self.min_dist_to_goal = gamestate["screen_info"]["dist_to_goal"]
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        if self.no_improve_counter >= 200:
            truncated = True
            reward_val -= 5

        if truncated:
            timesteps = gamestate["screen_info"]["time"] / 6
            reward_val -= 0.01 * (900 - timesteps)
        if done or truncated:
            self.min_dist_to_goal = float("inf")

        reward_val += (last_gamestate["screen_info"]["dist_to_goal"] - gamestate["screen_info"]["dist_to_goal"])*0.1

        if action[2]:
            reward_val -= 0.02
        
        return float(reward_val), done and not truncated, truncated

    def gamestate_to_observation(self, gamestate):
        # return gamestate

        observation = {}
        map_info = np.array(gamestate["screen_info"]["map_info"])[:, :, 0]  # (11, 21, 3) -> (11, 21)

        m0 = (map_info == 0)                                 # empty space
        m1 = (15 <= map_info) & (map_info <= 21)             # stairs, etc
        m2 = (map_info == 23)                                # exit
        m3 = np.isin(map_info, (13, 16))                     # platform
        m4 = ~(m0 | m1 | m2 | m3)                            # else -> ground
        multi_hot = np.stack([m0, m1, m2, m3, m4]).astype(np.uint8)
        # multi_hot = np.moveaxis(multi_hot, 0, -1)

        observation["map_info"] = multi_hot

        # with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
        #     np.set_printoptions(threshold=np.inf)
        #     f.write("TEST\n")
        #     f.write(f"map_info shape: {multi_hot.shape}\n")
        #     f.write(f"map_info:\n {multi_hot}\n")

        observation["char_state"] = np.int32(gamestate["player_info"]["char_state"])
        observation["can_jump"] = np.int32(int(gamestate["player_info"]["can_jump"]))

        return observation


if __name__ == '__main__':
    from datetime import datetime
    env = SpelunkyEnv(speedup=False, frames_per_step=4, reset_options={"ent_types_to_destroy":[600,601]})
    print("Game initialized")
    env.reset()
    print("Game started")

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