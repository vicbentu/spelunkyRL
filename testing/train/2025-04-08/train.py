# First training
# It was not used parallelism
# Used the observation space with all info (large discretes)

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback


from spelunkyRL.environments.complete_env import SpelunkyEnv

env = FlattenObservation(SpelunkyEnv(frames_per_step=6))

TIMESPTEPS = 5000
count = 0

checkpoint_callback = CheckpointCallback(
    save_freq=TIMESPTEPS,
    save_path="models/",
    name_prefix="ppo_spelunky",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=2,
    # device="cpu"
    tensorboard_log=r"D:\UserFolders\Desktop\Projects\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
)


while True:
    try:
        model.learn(
            total_timesteps=TIMESPTEPS,
            reset_num_timesteps=False,
            tb_log_name="8-4-25",
            callback=checkpoint_callback
        )
        count += 1
        # model.save(f"models/{TIMESPTEPS*count}_steps")
        print(f"Trained for {TIMESPTEPS*count} timesteps")
    except Exception as e:
        # Print current time
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()
        break