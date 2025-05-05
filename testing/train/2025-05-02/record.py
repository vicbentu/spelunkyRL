# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# import numpy as np

# from env import SpelunkyEnv
# from spelunkyRL.tools.save_replay import save_replay

# class RNNPolicyAgent:
#     def __init__(self, model, n_envs: int, deterministic: bool = False):
#         self.model = model
#         self.n_envs = n_envs
#         self.deterministic = deterministic
#         self.lstm_state = None
#         self.episode_start = np.ones((n_envs,), dtype=bool)

#     def predict(self, obs, dones: np.ndarray | None = None):
#         if dones is not None:
#             self.episode_start = dones.astype(bool)

#         actions, self.lstm_state = self.model.predict(
#             obs,
#             state=self.lstm_state,
#             episode_start=self.episode_start,
#             deterministic=self.deterministic
#         )

#         self.episode_start[:] = False
#         return actions

#     def reset(self):

#         self.lstm_state = None
#         self.episode_start.fill(True)

# if __name__ == "__main__":
#     entities_to_destroy = [600, 601] + list(range(219, 342))
#     def make_env():
#         def _init():
#             env = SpelunkyEnv(frames_per_step=6, speedup=False, reset_options={"ent_types_to_destroy":entities_to_destroy})
#             return env
#         return _init
#     n_envs = 1
#     env = DummyVecEnv([make_env() for _ in range(n_envs)])
#     env = VecNormalize.load(
#         r"testing\train\2025-04-30\models3\ppo_spelunky_vecnormalize_6062080_steps.pkl",
#         env,                 # the DummyVecEnv we just built
#     )
#     env.training = False
#     env.norm_reward = False

#     model = RecurrentPPO.load(
#         r"testing\train\2025-04-30\models3\ppo_spelunky_6062080_steps.zip",
#         env=env,
#         device="cuda",
#     )

#     policy_agent = RNNPolicyAgent(model,n_envs)

#     for _ in range(1):
#         save_replay(env, policy_agent, "test_vid.mp4", max_steps=60, fps=60)









from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

from env import SpelunkyEnv
from spelunkyRL.tools.save_replay import save_replay

class RNNPolicyAgent:
    def __init__(self, model, n_envs: int, deterministic: bool = False):
        self.model = model
        self.n_envs = n_envs
        self.deterministic = deterministic
        self.lstm_state = None
        self.episode_start = np.ones((n_envs,), dtype=bool)

    def predict(self, obs, dones: np.ndarray | None = None):
        if dones is not None:
            self.episode_start = dones.astype(bool)

        actions, self.lstm_state = self.model.predict(
            obs,
            state=self.lstm_state,
            episode_start=self.episode_start,
            deterministic=self.deterministic
        )

        self.episode_start[:] = False
        return actions

    def reset(self):

        self.lstm_state = None
        self.episode_start.fill(True)

if __name__ == "__main__":
    entities_to_destroy = [600, 601] + list(range(219, 342))
    def make_env():
        def _init():
            env = SpelunkyEnv(frames_per_step=6, speedup=False, reset_options={"ent_types_to_destroy":entities_to_destroy})
            return env
        return _init()
    n_envs = 1
    env = make_env()
    
    model = RecurrentPPO.load(
        r"testing\train\2025-04-30\models3\ppo_spelunky_6062080_steps.zip",
        env=env,
        device="cuda",
    )

    policy_agent = RNNPolicyAgent(model,n_envs)

    for _ in range(1):
        save_replay(env, policy_agent, "test_vid.mp4", max_steps=60, fps=60)