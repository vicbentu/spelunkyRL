from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import torch as th


# from spelunkyRL.environments.simple_environment import SpelunkyEnv
from dummy_environment import SpelunkyEnv

def make_env():
    def _init():
        env = SpelunkyEnv(
            frames_per_step=6,
            speedup=False,
            render_enabled=False,
            spelunky_dir=r"C:\TFG\Project\SpelunkyRL\Spelunky 2",
            playlunky_dir=r"C:\Users\vicbe\AppData\Local\spelunky.fyi\modlunky2\playlunky\nightly",
            console=False,

            # reset_options
            manual_control=True
        )
        return env
    return _init

if __name__ == "__main__":
    n_envs = 1
    env = SubprocVecEnv([make_env() for i in range(n_envs)])

    num_episodes_to_watch = 50
    episode_start = np.ones((n_envs,), dtype=bool)

    obs = env.reset()
    num_episodes = [0] * n_envs

    from datetime import datetime
    startTime = datetime.now()
    counter = 0

    while any(ep < num_episodes_to_watch for ep in num_episodes):

        counter += n_envs
        if counter % 100 == 0:
            print(f"{counter} frames have passed, FPS: {counter/(datetime.now()-startTime).total_seconds()}")

        actions = [env.action_space.sample() for _ in range(n_envs)]
        next_obs, rewards, dones, infos = env.step(actions)

        episode_start = dones.copy()
        for idx, done in enumerate(dones):
            if done:
                num_episodes[idx] += 1
                print(f"Episode {num_episodes[idx]} finished")
        obs = next_obs

    env.close()

