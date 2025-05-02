from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


from env_14_04_25 import SpelunkyEnv



def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6)
        env = Monitor(env)
        # env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(1)])
    # env = SubprocVecEnv(make_env())
    # env = make_env()()

    model_path = r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\15-04-25\models\ppo_spelunky_80_steps.zip"
    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log=r"logs"
    )

    # model.learn(
    #     total_timesteps=3000,
    #     reset_num_timesteps=False
    # )

    num_episodes_to_watch = 5

    

    for episode in range(num_episodes_to_watch):
        obs = env.reset()
        
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        
        print(f"Episode {episode + 1} finished. Total reward: {total_reward}")

    env.close()
