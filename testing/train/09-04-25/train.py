# Parallel training
# Used the observation space with all info (large discretes)

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import psutil
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# from spelunkyRL.environments.env_09_04_25 import SpelunkyEnv
from env_09_04_25 import SpelunkyEnv




def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6)
        env = Monitor(env)
        env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 700
    TIMESPTEPS = 3000

    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # env = DummyVecEnv([make_env() for _ in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESPTEPS,
        save_path="models/",
        name_prefix="ppo_spelunky",
        # save_replay_buffer=True,
        save_vecnormalize=True,
    )


    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=2,
    #     device="cuda",
    #     n_steps=TIMESPTEPS,
    #     tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
    # )
    model = PPO.load(
        path=str(r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\testing\train\09-04-25\models\ppo_spelunky_1848000_steps.zip"),
        env=env,
        device="cuda",
        tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs"
    )



    try:
        # model.learn(total_timesteps=TIMESPTEPS * TOTAL_LOOPS)
        model.learn(
            total_timesteps=TIMESPTEPS * TOTAL_LOOPS,
            tb_log_name="09-04-25",
            callback=[checkpoint_callback],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()