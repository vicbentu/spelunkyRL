# Run from latest model 29 #3
# Reduced jump penalization
# Videos didn't save properly, slowed too much the training
# 1 ->
# ent_coef = 0.02

# 2 -> 
# Removed VecNormalize to have raw env
# Without video recording



from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from spelunkyRL.tools.save_replay import save_replay
from env import SpelunkyEnv
import os
import numpy as np

# --- Environment factories ---
def make_train_env():
    def _init():
        entities_to_destroy = [600, 601] + list(range(219, 342))
        env = SpelunkyEnv(
            frames_per_step=6,
            speedup=True,
            reset_options={"ent_types_to_destroy": entities_to_destroy}
        )
        return Monitor(env)
    return _init

# Recording environment (no speedup)
def make_rec_env():
    def _init():
        entities_to_destroy = [600, 601] + list(range(219, 342))
        env = SpelunkyEnv(
            frames_per_step=6,
            speedup=False,
            reset_options={"ent_types_to_destroy": entities_to_destroy}
        )
        return Monitor(env)
    return _init

# RNN policy agent wrapper
def make_agent(model, n_envs=1, deterministic=False):
    class Agent:
        def __init__(self):
            self.model = model
            self.n_envs = n_envs
            self.deterministic = deterministic
            self.lstm_state = None
            self.episode_start = np.ones((n_envs,), dtype=bool)

        def predict(self, obs, dones=None):
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

    return Agent()

class SaveReplayCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix,
        save_vecnormalize,
        record_env,
        num_replays=5
    ):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_vecnormalize=save_vecnormalize
        )
        self.record_env = record_env
        self.num_replays = num_replays

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            steps = self.model.num_timesteps
            # Paths for model and VecNormalize
            # model_file = os.path.join(self.save_path, f"{self.name_prefix}_{steps}_steps.zip")
            # vec_file = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{steps}_steps.pkl")
            model_file = r"testing\train\2025-04-30\models3\ppo_spelunky_6062080_steps.zip"
            vec_file = r"testing\train\2025-04-30\models3\ppo_spelunky_vecnormalize_6062080_steps.pkl"

            # Reload VecNormalize stats
            self.record_env = VecNormalize.load(vec_file, self.record_env)
            self.record_env.training = False
            self.record_env.norm_reward = False
            # Save multiple replays
            for i in range(1, self.num_replays + 1):
                model = RecurrentPPO.load(
                    model_file,
                    env=self.record_env,
                    device=self.model.device
                )
                agent = make_agent(model, n_envs=1, deterministic=False)
                replay_path = os.path.join(self.save_path, f"replay_{steps}steps_{i}.mp4")
                save_replay(self.record_env, agent, replay_path, max_steps=500)
        return result

if __name__ == "__main__":
    TOTAL_LOOPS = 16000
    # TIMESTEPS = 512
    TIMESTEPS = 100

    # 8 envs for training
    train_env = SubprocVecEnv([make_train_env() for _ in range(2)])
    train_env = VecNormalize.load(
        r"testing\train\2025-04-30\models3\ppo_spelunky_vecnormalize_6062080_steps.pkl",
        train_env
    )

    # 1 env for recording
    rec_env = DummyVecEnv([make_rec_env()])
    rec_env = VecNormalize(rec_env)
    rec_env.training = False
    rec_env.norm_reward = False

    # Callback instance
    checkpoint_callback = SaveReplayCallback(
        # save_freq=TIMESTEPS * 40,
        save_freq=TIMESTEPS,
        save_path=r"testing\train\2025-05-02\models2",
        name_prefix="ppo_spelunky",
        record_env=rec_env,
        # num_replays=5
        num_replays=1

    )

    # Load last model
    model = RecurrentPPO.load(
        r"testing\train\2025-04-30\models3\ppo_spelunky_6062080_steps.zip",
        env=train_env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        ent_coef=0.01,
        gamma=0.97,
        # tensorboard_log=r"testing\tensorboardlogs"
    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            # tb_log_name="2025-05-02_1",
            callback=[checkpoint_callback],
            reset_num_timesteps=False
        )
    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        train_env.close()
