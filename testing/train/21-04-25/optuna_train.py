# tune_ppo_spelunky.py
"""
Optuna-driven hyper‑parameter search for PPO on the custom Spelunky RL environment.

Running this script will:
1. Launch an Optuna study (SQLite backed by default).
2. For each trial, create vectorised training and evaluation envs.
3. Build a PPO model that uses the custom SpelunkyFeaturesExtractor.
4. Train the agent for a short budget (you can tweak TOTAL_TIMESTEPS).
5. Evaluate the agent every EVAL_FREQ steps; the best mean
   episodic return observed during training is the trial objective.
6. After the study finishes, the script prints the best hyper‑parameter
   set and saves the Optuna DB + the best trained model parameters.

Usage
-----
$ python tune_ppo_spelunky.py \
    --n-trials 50 \
    --total-timesteps 500_000 \
    --study-name spelunky_ppo \
    --storage sqlite:///spelunky_ppo.db

Once the study is complete you can load the best params via
>>> import optuna
>>> study = optuna.load_study(study_name="spelunky_ppo", storage="sqlite:///spelunky_ppo.db")
>>> best_params = study.best_params

and retrain a final agent with a higher timestep budget.
"""
import argparse
import os
from datetime import datetime
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
#  Custom Env + Feature Extractor imports
# -----------------------------------------------------------------------------
from env_14_04_25 import SpelunkyEnv


class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):
    """Same extractor as in your training script (duplicated here for self‑containment)."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.map_height = 11
        self.map_width = 21
        self.n_input_channels = 117

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute flattened CNN size
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_input_channels, self.map_height, self.map_width)
            cnn_flat = self.map_cnn(dummy).shape[1]

        self.map_linear = nn.Linear(cnn_flat, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations["map_info"].float()
        b = x.shape[0]
        x = x.view(b, self.n_input_channels, self.map_height, self.map_width)
        return self.map_linear(self.map_cnn(x))


# -----------------------------------------------------------------------------
#  Helper functions
# -----------------------------------------------------------------------------

def make_env(seed: int = 0):
    """Factory needed for SubprocVecEnv."""

    def _init():
        env = SpelunkyEnv(frames_per_step=6, speedup=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines the search space for PPO + our extractor."""
    return {
        # PPO core params
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.98, step=0.02),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15),
        # Network architecture (shared by policy & value nets)
        "policy_kwargs": {
            "features_extractor_class": SpelunkyFeaturesExtractor,
            "net_arch": [trial.suggest_categorical("layer_size", [128, 256, 512])] * 2,
        },
    }


def objective(trial: optuna.Trial, total_timesteps: int, n_envs: int, eval_freq: int):
    # Seed per trial for reproducibility
    seed = trial.number

    # Training environments
    env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])

    # Evaluation environment (single‑process)
    eval_env = make_env(seed + 10)()

    # Sample hyper‑params
    hyperparams = sample_ppo_params(trial)

    # Build model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **hyperparams,
    )

    # Callback: stop if no improvement over 5 evals, each eval with 10 episodes
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=5,
        verbose=0,
    )
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq // n_envs,  # eval_freq uses env.step calls
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_train_callback,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # Use the best mean reward recorded by the eval callback
        mean_reward = eval_callback.best_mean_reward if hasattr(eval_callback, "best_mean_reward") else None
        # Fallback: evaluate directly if callback didn't run
        if mean_reward is None:
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    finally:
        env.close()
        eval_env.close()

    # Track best model
    trial.set_user_attr("mean_reward", mean_reward)
    # Save best model so far
    if trial.study.best_trial == trial:
        model.save("best_spelunky_model.zip")

    return mean_reward


# -----------------------------------------------------------------------------
#  Main entry point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--total-timesteps", type=int, default=300_000, help="Timesteps per trial")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="How often to run evaluations (env steps)")
    parser.add_argument("--study-name", type=str, default="ppo_spelunky", help="Optuna study name")
    parser.add_argument("--storage", type=str, default="sqlite:///ppo_spelunky.db", help="Optuna DB URI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Time‑stamped storage fallback if the default DB already exists.
    if args.storage.startswith("sqlite") and os.path.exists(args.storage.split("///")[-1]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.storage = f"sqlite:///ppo_spelunky_{ts}.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, args.total_timesteps, args.n_envs, args.eval_freq),
        n_trials=args.n_trials,
        n_jobs=1,  # GPU training is *not* thread‑safe; keep at 1 unless using CPU only
        show_progress_bar=True,
    )

    print("\n=== Study finished ===")
    print("Best trial:")
    print(f"  Number: {study.best_trial.number}")
    print(f"  Reward: {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # You can now reload the best model from best_spelunky_model.zip
