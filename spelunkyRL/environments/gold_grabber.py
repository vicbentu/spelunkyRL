"""
Gold Grabber Environment

A resource collection task where the agent must collect as much gold as possible within the time limit.

Observation Space:
- Terrain grid (11x21) of entity type IDs
- Gold value grid (11x21) showing the value of gold items at each position
- Character state (current animation/action state)
- Can jump (whether the player can currently jump)

Action Space:
- 3-action simplified space: [Movement X, Movement Y, Jump]

Reward Function:
- Reward proportional to gold collected each step (delta_money / 1000.0)
- Tracks total episode gold in info dict
- Truncation after 30 seconds

Notes:
- Destroys all enemies and traps at level start
- Gold values are mapped to grid positions for spatial awareness
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from spelunkyRL import SpelunkyRLEngine

# Map entity type IDs to their gold value
GOLD_VALUE_MAP = {
    495: 500,    # Gold nugget
    496: 1500,   # Gold bar
    497: 5000,   # Gold idol
    498: 800,    # Ruby
    499: 1200,   # Sapphire
    500: 1600,   # Emerald
    501: 500,    # Diamond
    502: 500,    # Large emerald
    503: 200,    # Large ruby
    504: 300,    # Large sapphire
    505: 400,    # Large diamond
    506: 100,    # Small gold nugget
}


class SpelunkyEnv(SpelunkyRLEngine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ################## ENV CHARACTERISTICS ##################

    observation_space = Dict({
        'map_info': Box(low=0, high=116, shape=(1, 11, 21), dtype=np.int32),
        'gold_info': Box(low=0, high=np.inf, shape=(1, 11, 21), dtype=np.int32),
        "char_state": Discrete(23),
        "can_jump": Discrete(2),
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
        reward_val = 0.0

        # Track episode gold
        current_money = gamestate["basic_info"]["money"]
        last_money = last_gamestate["basic_info"]["money"] if last_gamestate else current_money
        delta_money = current_money - last_money

        # Reward based on gold collected this step
        reward_val = delta_money / 1000.0

        # Update episode gold counter
        episode_gold = getattr(self, "episode_gold", 0)
        if delta_money > 0:
            episode_gold += delta_money
        self.episode_gold = episode_gold

        # Info tracking
        info["gold_delta"] = delta_money
        info["gold_collected"] = episode_gold

        # Time limit: 30 seconds
        if gamestate["basic_info"]["time"] >= 60 * 30:
            truncated = True

        # Reset episode gold on episode end
        if done or truncated:
            info["time"] = gamestate["basic_info"]["time"]
            info["total_gold_collected"] = self.episode_gold
            if truncated:
                info["success"] = False
            self.episode_gold = 0

        return float(reward_val), done or truncated, truncated, info

    def gamestate_to_observation(self, gamestate):
        observation = {}

        # Terrain grid
        observation["map_info"] = np.array(gamestate["map_info"])[np.newaxis, :, :]

        # Character state
        observation["char_state"] = np.int32(np.clip(gamestate["basic_info"]["char_state"], 0, 22))
        observation["can_jump"] = np.int32(int(gamestate["basic_info"]["can_jump"]))

        # Create gold value grid from entity positions
        gold_map = np.zeros((1, 11, 21), dtype=np.int32)
        x_rest = gamestate["basic_info"]["x_rest"]
        y_rest = gamestate["basic_info"]["y_rest"]

        for ent in gamestate["entity_info"]:
            gold_value = GOLD_VALUE_MAP.get(ent[4])
            if gold_value is None:
                continue

            # Convert entity position to grid coordinates
            x = int(ent[0] + x_rest + 10.5)
            y = 11 - int(ent[1] + y_rest + 5.5)

            if 0 <= x < 21 and 0 <= y < 11:
                gold_map[0, y, x] = min(
                    np.iinfo(np.int32).max,
                    gold_map[0, y, x] + gold_value
                )

        observation["gold_info"] = gold_map

        return observation
