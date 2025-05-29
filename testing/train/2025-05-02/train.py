# Run from latest model 29 #3
# Reduced jump penalization
# Videos didn't save properly, slowed too much the training
# 1 ->
# ent_coef = 0.02

# 2 -> 
# Removed VecNormalize to have raw env
# Without video recording

# 3 ->
# No video recording
# Started from scratch

# 4 ->
# Without LSTM
# ent_coef=0.02,
# gamma=0.97,

# 5 ->
# With LSTM
# Removed penalization for jumping
# ent_coef=0.02,
# gamma=0.98,
# Added char_state, can_jump


# 6 ->
# Changed reward function to penalize early truncation

# 7 -> crashed due to gym bug
# lstm_hidden_size=512
# n_lstm_layers=2,

# 8 -> same params, crashed again

# 9 -> same params, crashed again

# 10 -> same params, 

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder

import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from env import SpelunkyEnv
from spelunkyRL.tools.save_replay import save_replay

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

        n_flatten += 25 # char_state (from 0 to 22) + can_jump discrete(1)

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["map_info"].float()
        map_features = self.map_cnn(x)

        # ---- player info -------------------------------------------------------
        char_onehot = observations["char_state"].float().view(map_features.size(0), -1)
        jump_onehot = observations["can_jump"].float().view(map_features.size(0), -1)

        # with open(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\log.txt", "a") as f:
        #     f.write(str(char_onehot) + "\n")
        #     f.write(str(jump_onehot) + "\n")

        extra_feats = torch.cat([char_onehot, jump_onehot], dim=1)
        all_feats   = torch.cat([map_features, extra_feats], dim=1)

        return self.linear(all_feats)

def make_env(i):
    def _init():
        entities_to_destroy = [600,601] + list(range(219, 342)) + list(range(899, 906))
        # if i == 0:
        #     env = SpelunkyEnv(frames_per_step=6, speedup=True, reset_options={"ent_types_to_destroy":entities_to_destroy}, render_enabled=True)
        #     env = RecordVideo(
        #         env,
        #         video_folder=r"testing\train\2025-05-02\videos9",
        #         episode_trigger=lambda episode_id: episode_id % 10 == 0,
        #         fps=10,
        #         gc_trigger=lambda episode_id: episode_id % 10 == 0,
        #     )
        # else:
        env = SpelunkyEnv(frames_per_step=6, speedup=True, reset_options={"ent_types_to_destroy":entities_to_destroy}, render_enabled=False)

        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 16000*2
    TIMESTEPS = 2048 

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    # env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=15.)
    # env = VecNormalize.load(r"testing\train\2025-04-30\models3\ppo_spelunky_vecnormalize_6062080_steps.pkl", env)

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS*40,
        save_path=r"testing\train\2025-05-02\models10",
        name_prefix="ppo_spelunky",
    )

    # model = PPO(
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        # policy="MultiInputPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        tensorboard_log=r"testing\tensorboardlogs",
        policy_kwargs=dict(
            features_extractor_class=SpelunkyFeaturesExtractor,
            features_extractor_kwargs={},
            net_arch=[64, 64],
            lstm_hidden_size=512,
            n_lstm_layers=2,
        ),

        # ent_coef=LinearSchedule(0.02, 0.0, TOTAL_LOOPS * TIMESTEPS),
        # learning_rate=LinearSchedule(0.0005, 0.0001, TOTAL_LOOPS * TIMESTEPS),
        # clip_range=LinearSchedule(0.2, 0.05, TOTAL_LOOPS * TIMESTEPS),
        # ent_coef     = 0.02,
        # gamma=0.99,

        # learning_rate = get_linear_fn(0.0005, 0.0001, TOTAL_LOOPS * TIMESTEPS),
        # clip_range    = get_linear_fn(0.2,    0.05,  TOTAL_LOOPS * TIMESTEPS),
        # learning_rate = get_linear_fn(0.0005, 0.0001, 1),
        # clip_range    = get_linear_fn(0.2,    0.05, 1),

        ent_coef=0.02,
        gamma=0.98,
    )

    # model = RecurrentPPO.load(
    #     r"testing\train\2025-04-30\models3\ppo_spelunky_6062080_steps.zip",
    #     env=env,
    #     verbose=2,
    #     device="cuda",
    #     n_steps=TIMESTEPS,
    #     ent_coef=0.01,
    #     gamma=0.97,
    #     tensorboard_log=r"testing\tensorboardlogs",
    # )


    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            tb_log_name="2025-05-02_10",
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