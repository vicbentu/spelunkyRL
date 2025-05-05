# Parallel training
# Reduced environment
# Maked a model to learn to throw EXACTLY 2 bombs

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



from env_14_04_25 import SpelunkyEnv

# class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):

#     def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
#         super().__init__(observation_space, features_dim=features_dim)

#         ############ CNN for map info
#         self.map_height = 11
#         self.map_width = 21
#         self.n_input_channels = 117

#         self.map_cnn = nn.Sequential(
#             nn.Conv2d( self.n_input_channels, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         with torch.no_grad():
#             dummy = torch.zeros(
#                 1, self.n_input_channels, self.map_height, self.map_width
#             )
#             cnn_flat = self.map_cnn(dummy).shape[1]

#         self.map_linear = nn.Linear(cnn_flat, 128)

#         self.fuse_linear = nn.Sequential(
#             nn.Linear(128+1, features_dim),
#             nn.ReLU()
#         )



#     def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

#         x = observations["map_info"].float()      # (B, 27027)
#         # reshape to (batch, channels, H, W) = (B, 117, 11, 21)
#         B = x.shape[0]
#         x = x.view(B,
#                    self.n_input_channels,
#                    self.map_height,
#                    self.map_width)
#         map_features = self.map_linear(self.map_cnn(x))                # (B, features_dim)


#         bombs = observations["bombs"].float().view(B, 1)

#         return self.fuse_linear(torch.cat([map_features, bombs], dim=1))

class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)

        ############ CNN for map info
        self.map_height = 11
        self.map_width = 21
        self.n_input_channels = 117

        self.map_cnn = nn.Sequential(
            nn.Conv2d( self.n_input_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, self.n_input_channels, self.map_height, self.map_width
            )
            cnn_flat = self.map_cnn(dummy).shape[1]

        self.map_linear = nn.Linear(cnn_flat, 128)

        self.fuse_linear = nn.Sequential(
            nn.Linear(128+1, features_dim),
            nn.ReLU()
        )


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        x = observations["map_info"].float()      # (B, 27027)
        # reshape to (batch, channels, H, W) = (B, 117, 11, 21)
        B = x.shape[0]
        x = x.view(B,
                   self.n_input_channels,
                   self.map_height,
                   self.map_width)
        map_features = self.map_linear(self.map_cnn(x))                # (B, features_dim)


        bombs = observations["bombs"].float().view(B, 1)

        return self.fuse_linear(torch.cat([map_features, bombs], dim=1))


def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=True)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 5000
    TIMESTEPS = 300

    num_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS,
        save_path="models/",
        name_prefix="ppo_spelunky",
        save_vecnormalize=True,
    )


    # model = PPO(
    #     policy="MultiInputPolicy",
    #     env=env,
    #     verbose=2,
    #     device="cpu",
    #     n_steps=TIMESTEPS,
    #     # tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
    #     policy_kwargs=dict(
    #         # features_extractor_class=SpelunkyFeaturesExtractor,
    #         features_extractor_kwargs={},
    #         net_arch=[128, 128],
    #     ),

    #     # HYPERPARAMETERS
    #     learning_rate=3e-3,       # or bump to 1e‑3 / 3e‑3 if you want more aggressive updates
    #     n_epochs=10,              # number of times to reuse each rollout
    #     gamma=0.95,               # effective horizon (lower -> short-term focus)
    #     gae_lambda=0.9,           # GAE parameter (lower -> more bias, higher -> more variance)
    #     ent_coef=0.01,            # entropy coefficient (lower -> less exploration)
    #     clip_range = 0.3,         # PPO clipping parameter (lower -> more conservative)

    # )

    model  = PPO.load(
        path = r"C:\Users\vicbe\Desktop\TFG\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\17-04-25\models\ppo_spelunky_50400_steps.zip",
        env=env,
        verbose=2,
        device="cpu",
        n_steps=TIMESTEPS,
    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            # tb_log_name="17-04-25",
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