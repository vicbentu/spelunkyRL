from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import torch as th


from env import SpelunkyEnv

def make_env(i):
    def _init():
        entities_to_destroy = [600,601] + list(range(219, 342))
        env = SpelunkyEnv(frames_per_step=6, speedup=False, reset_options={"ent_types_to_destroy":entities_to_destroy})

        # Test
        if i == 0:
            env = RecordVideo(
                env,
                video_folder=r"videos",
                episode_trigger= lambda x: x % 10 == 0, # every 10 episodes,
                fps=10
            )
        # 

        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    n_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    # env = VecNormalize.load(r"testing\train\2025-05-02\models\ppo_spelunky_vecnormalize_8519680_steps.pkl", env)

    # when testing
    # env.training = False
    # env.norm_reward = False


    model_path = r"testing\train\2025-05-02\models2\ppo_spelunky_7864320_steps.zip"
    model = RecurrentPPO.load(
        model_path,
        env=env,
        # tensorboard_log=r"logs"
    )

    num_episodes_to_watch = 20
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

    while any(ep < num_episodes_to_watch for ep in num_episodes):
        # actions, _ = model.predict(obs, deterministic=True)
        actions, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=False
        )

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
                # lstm_state[idx] = None
        obs = next_obs

    env.close()

