# 1 -> reward incorrect, didn't know you can get to the goal

# 2 -> Experiment all: cnn, reward for getting closer, LSTM, 

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

import torch
import gymnasium as gym
import torch.nn as nn
from typing import Dict
import numpy as np

from spelunkyRL.environments.simple_environment import SpelunkyEnv

##################### LOGGER #####################

class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_successes.append(info.get("success", False))
        return True

    def _on_rollout_end(self) -> None:
        if self.episode_successes:
            # Calculate success rate over recent episodes
            recent_success_rate = 100 * np.mean(self.episode_successes[-100:])
            self.logger.record("custom/success_rate", recent_success_rate)
            if self.verbose:
                print(f"Success rate over last 100 episodes: {recent_success_rate:.2f}%")


##################### FEATURES EXTRACTOR #####################

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

        extra_feats = torch.cat([char_onehot, jump_onehot], dim=1)
        all_feats   = torch.cat([map_features, extra_feats], dim=1)

        return self.linear(all_feats)

def make_env(i):
    def _init():
        env = SpelunkyEnv(
            spelunky_dir=r"C:\TFG\Project\SpelunkyRL\Spelunky 2",
            playlunky_dir=r"C:\Users\vicbe\AppData\Local\spelunky.fyi\modlunky2\playlunky\nightly",
            frames_per_step=6,
            speedup=True,
            manual_control=False,
            god_mode=True
        )
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 16000*2
    TIMESTEPS = 2048

    num_envs = 6
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS*40,
        save_path=r"testing\train\2025-05-21\models2",
        name_prefix="ppo_spelunky",
    )

    model = RecurrentPPO(
    # model = PPO(
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
            # lstm_hidden_size=512,
            # n_lstm_layers=2,
        ),

        ent_coef=0.02,
        gamma=0.98,
    )


    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            tb_log_name="2025-05-21_2",
            callback=[
                checkpoint_callback, SuccessRateCallback(verbose=1)
            ],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()