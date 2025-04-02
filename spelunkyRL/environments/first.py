from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, Tuple
from spelunkyRL import SpelunkyEnv
import numpy as np

# Entities
entity_space = Tuple((
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

    # 'entities': Tuple(tuple(entity_space for _ in range(150)))
})


def reward(observation, last_observation):
    reward_val = 0
    if observation["health"] <= 0:
        reward_val -= 50
    else:
        hp_loss = last_observation["health"] - observation["health"]
        reward_val -= hp_loss * 5

    # y_diff = observation["y"] - last_observation["y"]
    # reward_val -= y_diff * 1

    if observation["win"]:
        reward_val += 100

    return float(reward_val)


def gamestate_to_observation(observation):
    for key, space in observation_space.spaces.items():
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

    # observation["entities"] = tuple(transformed)
    if "entities" in observation:
        del observation["entities"]
    return observation
