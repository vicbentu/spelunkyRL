# Parallel training
# Added instrinsic curiosity model
# Used the observation space with all info (large discretes)

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import psutil
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC
import torch as th

from rllte.xplore.reward import E3B
import numpy as np



from env_09_04_25 import SpelunkyEnv


class RLeXploreWithOnPolicyRL(BaseCallback):
    def __init__(self, irs, action_space, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.action_space = action_space

        # Prepare dimensions for action encoding
        self.action_dims = action_space.nvec
        self.total_action_dim = np.sum(self.action_dims)

    def init_callback(self, model) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def encode_actions(self, actions):
        original_shape = actions.shape
        actions_flat = actions.view(-1, len(self.action_dims))

        encoded = []
        for idx, dim in enumerate(self.action_dims):
            one_hot = th.nn.functional.one_hot(actions_flat[:, idx].long(), num_classes=dim)
            encoded.append(one_hot)

        encoded_actions = th.cat(encoded, dim=-1).float()
        encoded_actions = encoded_actions.view(*original_shape[:-1], -1)
        return encoded_actions

    def _on_step(self) -> bool:
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # Encode actions properly
        encoded_actions = self.encode_actions(actions)

        # watch the interaction with encoded actions
        self.irs.watch(observations, encoded_actions, rewards, dones, dones, next_observations)
        return True

    def _on_rollout_end(self) -> None:
        obs = th.as_tensor(self.buffer.observations)
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])

        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)

        # Encode actions properly
        encoded_actions = self.encode_actions(actions)

        intrinsic_rewards = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=encoded_actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )

        # Add intrinsic rewards to advantages and returns
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()

def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6)
        env = Monitor(env)
        env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 700
    # TIMESPTEPS = 3000
    TIMESTEPS = 300

    num_envs = 2
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # env = DummyVecEnv([make_env() for _ in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS,
        save_path="models/",
        name_prefix="ppo_spelunky",
        # save_replay_buffer=True,
        save_vecnormalize=True,
    )


    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        # tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
    )



    try:
        # model.learn(total_timesteps=TIMESPTEPS * TOTAL_LOOPS)
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            # tb_log_name="10-04-25",
            callback=[
                checkpoint_callback,
                RLeXploreWithOnPolicyRL(E3B(env, device="cpu"), action_space=env.action_space)
            ],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()