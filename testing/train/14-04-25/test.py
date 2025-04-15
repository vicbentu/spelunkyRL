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

    model_path = r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\14-04-25\models\ppo_spelunky_4176000_steps.zip"
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

    
    def ensure_batch_dimension(obs):
        # If observation is a dict, ensure every value has a batch dim.
        if isinstance(obs, dict):
            new_obs = {}
            for key, value in obs.items():
                # If it's a numpy array or tensor and has 1 less dimension than expected, add a batch dimension.
                if hasattr(value, "ndim") and value.ndim == 1:
                    # Use None to add a new axis at the beginning
                    new_obs[key] = value[None, ...]
                else:
                    new_obs[key] = value
            return new_obs
        # If it's a numpy array or tensor and unbatched, add batch dimension.
        if hasattr(obs, "ndim") and obs.ndim == 1:
            return obs[None, ...]
        return obs

    # In your evaluation loop
    for episode in range(num_episodes_to_watch):
        obs = env.reset()
        # Fix observation for proper dimensions if necessary
        obs = ensure_batch_dimension(obs)
        
        # Do an initial step if needed
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = ensure_batch_dimension(obs)
        
        done = False
        total_reward = 0
        while not done:
            print("STEP")
            # Ensure the observation is correctly batched before each prediction
            obs = ensure_batch_dimension(obs)
            action, _ = model.predict(obs, deterministic=True)
        
            obs, reward, done, info = env.step(action)
            obs = ensure_batch_dimension(obs)
            total_reward += reward[0]
        
        print(f"Episode {episode + 1} finished. Total reward: {total_reward}")

    env.close()
