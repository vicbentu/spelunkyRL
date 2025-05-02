from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize



import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np
import torch as th


from env_14_04_25 import SpelunkyEnv



def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=False)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    n_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize.load(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\22-04-25\models\ppo_spelunky_vecnormalize_8064000_steps.pkl", env)

    # when testing
    env.training = False
    env.norm_reward = False


    model_path = r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\22-04-25\models\ppo_spelunky_8064000_steps.zip"
    model = PPO.load(
        model_path,
        env=env,
        # tensorboard_log=r"logs"
    )

    num_episodes_to_watch = 20
    obs = env.reset()
    num_episodes = [0] * n_envs

    # let's run until each env has seen 5 episodes
    while any(ep < num_episodes_to_watch for ep in num_episodes):
        actions, _ = model.predict(obs, deterministic=True)
        next_obs, rewards, dones, infos = env.step(actions)

        # ------ probability inspection --------------------------------------
        def action_probs(dist):
            if isinstance(dist.distribution, list):          # MultiDiscrete
                return [cat.probs.detach().cpu().numpy()
                        for cat in dist.distribution]        # len == n_subspaces
            else:                                            # plain Discrete or MultiBinary
                return dist.distribution.probs.detach().cpu().numpy()
            
        obs_tensor, _ = model.policy.obs_to_tensor(next_obs)
        with th.no_grad():
            dist  = model.policy.get_distribution(obs_tensor)
            probs = action_probs(dist)
        # print(probs)
        print(rewards)
        # --------------------------------------------------------------------


        # for each parallel env that just finished:
        for idx, done in enumerate(dones):
            if done:
                num_episodes[idx] += 1
                # print(f">>> Sub‑env {idx} finished episode {num_episodes[idx]}")

        obs = next_obs

    env.close()

