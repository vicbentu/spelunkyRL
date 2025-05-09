# Only ground, god mode
# New cnn with reduced size
# Used LSTM

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn

import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



from env import SpelunkyEnv

class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)

        ############ CNN for map info
        self.map_height = 11
        self.map_width = 21
        self.n_input_channels = 5

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.map_cnn(torch.zeros(1, self.n_input_channels, self.map_height, self.map_width)).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["map_info"].float()
        # x = x.permute(0, 3, 1, 2)

        # with torch.no_grad():
        #     x_np = x.cpu().numpy()
        #     np.set_printoptions(threshold=np.inf)  # disable truncation
        #     with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
        #         f.write(f"Type: {type(x_np)}\n")
        #         f.write(f"Shape: {x_np.shape}\n")
        #         f.write(f"map_info: {x_np}\n")

        map_features = self.linear(self.map_cnn(x))

        return map_features

def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=True, reset_options={"ent_types_to_destroy":[600,601]})
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 40000
    TIMESTEPS = 512

    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=15.)


    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS*40,
        save_path=r"testing\train\29-04-25\models2",
        name_prefix="ppo_spelunky",
        save_vecnormalize=True,
    )

    # model = PPO(
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        # tensorboard_log=r"testing\tensorboardlogs",
        policy_kwargs=dict(
            features_extractor_class=SpelunkyFeaturesExtractor,
            features_extractor_kwargs={},
            net_arch=[64, 64],
        ),

        ent_coef=0.02,
        gamma=0.98,

    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            # tb_log_name="29-04-25_1",
            callback=[
                checkpoint_callback,
            ],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()