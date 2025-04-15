from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple
import numpy as np

from spelunkyRL import SpelunkyRLEngine

class SpelunkyEnv(SpelunkyRLEngine):

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        'health': Discrete(100),
        'bombs': Discrete(100),
        'ropes': Discrete(100),
        'money': Box(low=0, high=2e7, shape=(1,), dtype=np.float32),

        'vel_x': Box(low=-10, high=10, shape=(1,), dtype=np.float32),
        'vel_y': Box(low=-10, high=10, shape=(1,), dtype=np.float32),
        'face_left': Discrete(2),
        'powerups': MultiBinary(18),
        # 'holding': Box(low=0, high=915, shape=(20, 35)),
        # 'back': Box(low=-1, high=915, shape=(20, 35)),
        'holding_type_player': Discrete(916),
        'back_item': Discrete(916),

        'world': Discrete(99),
        'level': Discrete(100),
        'theme': Discrete(18),
        'time': Box(low=0, high=np.inf, shape=(), dtype=np.int64),
        'win': Discrete(2),

        'entities': Tuple(tuple(
            Tuple((
                Tuple((
                    Discrete(916),
                    Discrete(3),
                    Discrete(8),
                    Discrete(916),
                )),
                Box(
                    low=np.array([-17., -9.0, -10.0, -10.0]),
                    high=np.array([17., 9.0, 10.0, 10.0]),
                    shape=(4,),
                    dtype=np.float32,
                ),
            ))
            for _ in range(150)))            
    })


    def reward_function(self, observation, last_observation):
        return 0
        reward_val = 0
        if observation["health"] <= 0:
            reward_val -= 50
        else:
            hp_loss = last_observation["health"] - observation["health"]
            reward_val -= hp_loss * 5


        if observation["win"]:
            reward_val += 100

        return float(reward_val)


    def gamestate_to_observation(self, observation):
        return observation
        for key, space in self.observation_space.spaces.items():
            if isinstance(space, Box) and key in observation:
                # Wrap scalar values into an array if the expected shape is non-empty
                if np.isscalar(observation[key]):
                    observation[key] = np.array([observation[key]], dtype=np.float32)
                else:
                    observation[key] = np.array(observation[key], dtype=np.float32)

        transformed = []
        for ent in observation["entities"]:
            x, y, vx, vy = float(ent[0]), float(ent[1]), float(ent[2]), float(ent[3])

            type_id     = int(ent[4])
            flip        = int(ent[5]+1) 
            supertype   = int(ent[6])
            held        = int(ent[7])

            discrete_tuple = (type_id, flip, supertype, held)
            float_array    = [x, y, vx, vy]

            transformed.append((discrete_tuple, float_array))

        if len(transformed) < 150:
            zeros = ((0, 0, 0, 0), [0.0, 0.0, 0.0, 0.0])
            while len(transformed) < 150:
                transformed.append(zeros)
        else:
            transformed = transformed[:150]

        observation["entities"] = tuple(transformed)
        # if "entities" in observation:
        #     del observation["entities"]
        return observation

if __name__ == '__main__':
    from datetime import datetime
    env = SpelunkyEnv()
    env.reset()

    startTime = datetime.now()

    counter = 0
    while True:
        counter += 1
        if counter % 60 == 0:
            print(f"{counter/60} seconds have passed, FPS: {counter/(datetime.now()-startTime).total_seconds()}")
        observation, reward, _, done, _ = env.step(env.action_space.sample())
        if done:
            print("Game Over")
            env.reset()