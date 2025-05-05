# Parallel training
# Used cnn for map info
# Added distance to goal
# Used godmode
# Only movement actions enabled
# 1st my hyperparameters
# 2nd default hyperparameters

# changed hyperparameters on step 6464000

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
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



from env_14_04_25 import SpelunkyEnv

class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)

        ############ CNN for map info
        self.map_height = 11
        self.map_width = 21
        self.n_input_channels = 117

        # self.map_cnn = nn.Sequential(
        #     nn.Conv2d( self.n_input_channels, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

        #     nn.MaxPool2d(2), # (256, 11, 21) -> (256, 5, 10)

        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

            
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        # )

        C, H, W = 117, 11, 21
        self.map_cnn = nn.Sequential(
            nn.Conv2d(C,   32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1, self.n_input_channels, self.map_height, self.map_width
            )
            cnn_flat = self.map_cnn(dummy).shape[1]

        self.map_linear = nn.Linear(cnn_flat, features_dim)


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["map_info"].long()        # (B,11,21)

        B, H, W = x.shape                            # just for clarity
        x = F.one_hot(x, num_classes=117)            # (B,H,W,117)
        x = x.permute(0, 3, 1, 2).float()            # (B,117,H,W)


        # with torch.no_grad():
            # labels = x.cpu().numpy().argmax(axis=1)[0]   # o x.cpu().argmax(1)[0].numpy()
            # lines = "\n".join(" ".join(f"{v:3d}" for v in row)   # :3d = ancho 3, alineado
            #                 for row in labels)
            # with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
            #     f.write(f"map_info shape: {labels.shape}\n")
            #     f.write("map_info:\n")
            #     f.write(lines + "\n")

            # x_np = x.cpu().numpy()
            # np.set_printoptions(threshold=np.inf)  # disable truncation
            # with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
            #     f.write(f"map_info: {x_np}\n")


        map_features = self.map_linear(self.map_cnn(x))                # (B, features_dim)

        return map_features

def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=True)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 3750
    TIMESTEPS = 4000

    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=15.)


    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS,
        save_path="models5/",
        name_prefix="ppo_spelunky",
        save_vecnormalize=True,
    )


    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
        policy_kwargs=dict(
            features_extractor_class=SpelunkyFeaturesExtractor,
            features_extractor_kwargs={},
            net_arch=[128, 128],
        ),

        # HYPERPARAMETERS test 1
        # learning_rate=3e-3,
        # n_epochs=10,
        # gamma=0.95,               # effective horizon (lower -> short-term focus)
        # gae_lambda=0.9,           # GAE parameter (lower -> more bias, higher -> more variance)
        # ent_coef=0.01,            # entropy coefficient (lower -> less exploration)
        # clip_range = 0.3,         # PPO clipping parameter (lower -> more conservative)

        # learning_rate=1e-4,          # was 3e-4?
        # clip_range=0.15,             # was 0.2?
        # clip_range_vf=0.15,          # optional
        # ent_coef=0.02,               # was 0.0 or 0.01
        # vf_coef=1.0,                 # weight value loss a bit more
        # gamma=0.995,
        # batch_size=1000

        # --- tuned hyper-parameters ---
        learning_rate = get_linear_fn(start=2.5e-4, end=0.0, end_fraction=1.0),
        clip_range = get_linear_fn(start=0.15, end=0.0, end_fraction=1.0),
        clip_range_vf=None,                      # let critic move freely
        ent_coef=0.02,                           # keep exploration
        vf_coef=0.5,                             # curb critic over-fit
        n_epochs=4,                              # fewer passes over batch
        batch_size=1000,                         # minibatch 1/4 of rollout
        target_kl=0.009,                         # early stop if KL drifts
        gamma=0.995,

    )

    # model  = PPO.load(
    #     path = r"C:\Users\vicbe\Desktop\TFG\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\17-04-25\models\ppo_spelunky_50400_steps.zip",
    #     env=env,
    #     verbose=2,
    #     device="cpu",
    #     n_steps=TIMESTEPS,
    # )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            tb_log_name="21-04-25_4",
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