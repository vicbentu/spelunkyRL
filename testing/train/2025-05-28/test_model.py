from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import torch as th


from spelunkyRL.environments.get_to_exit import SpelunkyEnv

def make_env(i):
    def _init():
        env = SpelunkyEnv(
            spelunky_dir=r"C:\TFG\Project\SpelunkyRL\Spelunky 2",
            playlunky_dir=r"C:\Users\vicbe\AppData\Local\spelunky.fyi\modlunky2\playlunky\nightly",
            frames_per_step=6,
            speedup=False,
            manual_control=False,
            god_mode=True
        )
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    n_envs = 1
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])


    model_path = r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\2025-05-21\models2\ppo_spelunky_2949120_steps.zip"
    model = RecurrentPPO.load(
        model_path,
        env=env,
        # tensorboard_log=r"logs"
    )

    num_episodes_to_watch = 50
    lstm_state = None
    episode_start = np.ones((n_envs,), dtype=bool)

    obs = env.reset()
    num_episodes = [0] * n_envs

    ###### TRYING THINGS
    # model = RecurrentPPO.load(
    #     model_path,
    #     env=env,
    #     device="cuda",
    #     # tensorboard_log=r"logs",
    #     n_steps=2000,

    #     ent_coef=0.02,                           # keep exploration
    #     gamma=0.98,
    # )
    # model.learn(
    #     total_timesteps=2000*1000,
    #     reset_num_timesteps=False
    # )
    #######

    from datetime import datetime
    startTime = datetime.now()
    counter = 0

    while any(ep < num_episodes_to_watch for ep in num_episodes):

        counter += n_envs
        if counter % 100 == 0:
            print(f"{counter} frames have passed, FPS: {counter/(datetime.now()-startTime).total_seconds()}")


        # actions = [env.action_space.sample() for _ in range(n_envs)]
        actions, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True
        )

        # one in a 2000 chance to wait for a second
        # if np.random.randint(2000) == 0:
        #     import time
        #     print("Sleeping for 2 seconds")
        #     time.sleep(2)
        next_obs, rewards, dones, infos = env.step(actions)

        # # ------ probability inspection --------------------------------------
        # def action_probs(dist):
        #     if isinstance(dist.distribution, list):          # MultiDiscrete
        #         return [cat.probs.detach().cpu().numpy()
        #                 for cat in dist.distribution]        # len == n_subspaces
        #     else:                                            # plain Discrete or MultiBinary
        #         return dist.distribution.probs.detach().cpu().numpy()
            
        # obs_tensor, _ = model.policy.obs_to_tensor(next_obs)
        # with th.no_grad():
        #     dist  = model.policy.get_distribution(
        #         obs_tensor,
        #         lstm_state,        # ← new hidden state returned by predict()
        #         episode_start
        #     )
        #     probs = action_probs(dist)
        # # print(probs)
        # print(rewards)
        # # --------------------------------------------------------------------

        episode_start = dones.copy()
        for idx, done in enumerate(dones):
            if done:
                num_episodes[idx] += 1
                print(f"Episode {num_episodes[idx]} finished")
                # lstm_state[idx] = None
        obs = next_obs

    env.close()

