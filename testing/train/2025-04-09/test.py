from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import psutil
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from env_09_04_25 import SpelunkyEnv


def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=False)
        env = Monitor(env)
        env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 700
    TIMESPTEPS = 3000

    num_envs = 1
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # env = DummyVecEnv([make_env() for _ in range(num_envs)])

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=TIMESPTEPS,
    #     save_path="models/",
    #     name_prefix="ppo_spelunky",
    #     # save_replay_buffer=True,
    #     save_vecnormalize=True,
    # )


    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=2,
    #     device="cuda",
    #     n_steps=TIMESPTEPS,
    #     # tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
    # )
    model = PPO.load(
        path=str(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\09-04-25\models\ppo_spelunky_120000_steps.zip"),
        env=env,
        device="cuda",
        # tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs"
    )

    try:
        # model.learn(total_timesteps=TIMESPTEPS * TOTAL_LOOPS)
        model.learn(
            total_timesteps=TIMESPTEPS * TOTAL_LOOPS,
            # tb_log_name="09-04-25",
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()































# # from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# # import gymnasium as gym
# # import numpy as np

# # class FlattenEntitiesWrapper(gym.ObservationWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
# #         # Get the original entities spaces from the Dict observation space.
# #         # Assume that 'entities' is defined as a list of identical Dict spaces.
# #         entities_spaces = self.observation_space.spaces['entities']
# #         self.num_entities = len(entities_spaces)
# #         # Assume every entity has the same structure; store one for flattening.
# #         self.entity_space = entities_spaces[0]
        
# #         # Determine the flattened dimension for one entity.
# #         entity_flat_dim = gym.spaces.flatdim(self.entity_space)
# #         total_flat_dim = entity_flat_dim * self.num_entities
        
# #         # Update the observation space for 'entities' to be a single Box.
# #         # (For simplicity, we use -infty/infty; you may compute bounds if needed.)
# #         self.observation_space.spaces['entities'] = gym.spaces.Box(
# #             low=-np.inf,
# #             high=np.inf,
# #             shape=(total_flat_dim,),
# #             dtype=np.float32
# #         )

# #     def observation(self, observation):
# #         flat_entities = []
# #         # Flatten each entity dictionary in the list.
# #         for entity in observation['entities']:
# #             flat_entity = gym.spaces.flatten(self.entity_space, entity)
# #             flat_entities.append(flat_entity)
# #         # Concatenate all flattened entities into one 1D array.
# #         observation['entities'] = np.concatenate(flat_entities)
# #         return observation





# # from stable_baselines3 import PPO
# # from gymnasium.wrappers import FlattenObservation
# # from stable_baselines3.common.callbacks import CheckpointCallback


# # from spelunkyRL.environments.env_09_04_25 import SpelunkyEnv

# # env = FlattenObservation(SpelunkyEnv(frames_per_step=6))
# # # env = SpelunkyEnv(speedup=True, frames_per_step=6)
# # # env = FlattenEntitiesWrapper(SpelunkyEnv(speedup=True, frames_per_step=6))

# # # model = PPO.load(
# # #     "models/190000_steps.zip",
# # #     device="cpu"
# # # )
# # # model.set_env(env)

# # model = PPO(
# #     policy="MlpPolicy",
# #     env=env,
# #     verbose=2,
# # )


# # model.learn(total_timesteps=10000, reset_num_timesteps=False)


# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecTransposeImage  # If you're using image obs
# from stable_baselines3.common.vec_env import VecFrameStack
# from gymnasium.wrappers import FlattenObservation
# import numpy as np


# from spelunkyRL.environments.env_09_04_25 import SpelunkyEnv


# def make_env():
#     def _init():
#         env = SpelunkyEnv(frames_per_step=6)
#         env = FlattenObservation(env)
#         return env
#     return _init

# if __name__ == "__main__":
#     num_envs = 1
#     env = SubprocVecEnv([make_env() for _ in range(num_envs)])
#     # env = FlattenObservation(SpelunkyEnv(frames_per_step=6))

#     model = PPO(
#         policy="MlpPolicy",
#         env=env,
#         verbose=2,
#         device="cpu",
#     )

#     try:
#         model.learn(total_timesteps=1000, reset_num_timesteps=False)
#         env.close()
#     except Exception as e:
#         env.close()
#         print(f"Error: {e}")