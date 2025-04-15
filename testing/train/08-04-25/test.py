from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback


from spelunkyRL.environments.complete_env import SpelunkyEnv

# env = FlattenObservation(SpelunkyEnv(frames_per_step=6))
env = SpelunkyEnv(speedup=True, frames_per_step=6)

# model = PPO.load(
#     "models/190000_steps.zip",
#     device="cpu"
# )
# model.set_env(env)

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=2,
    # device="cpu"
)


model.learn(total_timesteps=10000, reset_num_timesteps=False)