# Parallel training


from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



from env_14_04_25 import SpelunkyEnv

########### FEATURE EXTRACTOR ###########

class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Example custom feature extractor that:
      - Embeds large discrete fields (like 'type_id', 'holding')
      - Concatenates with continuous parts (like velocity, health, bombs, etc.)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128, embedding_dim=16):
        super().__init__(observation_space, features_dim=features_dim)

        ############ CNN for map info
        
        self.map_height = 11
        self.map_width = 21

        self.embedding_floor_type = nn.Embedding(
            num_embeddings=117,
            embedding_dim=embedding_dim
        )
        self.cnn = nn.Sequential(
            # input, output, 
            nn.Conv2d(embedding_dim + 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Get the flattened CNN size
        with torch.no_grad():
            dummy_in = torch.zeros(1, embedding_dim + 2, self.map_height, self.map_width)
            dummy_out = self.cnn(dummy_in)
            cnn_out_dim = dummy_out.shape[1]

        self.cnn_linear = nn.Linear(cnn_out_dim, 128)
        self.cnn_output_dim = 128


        ############ MLP for entities

        self.embedding_type_id = nn.Embedding(
            num_embeddings=507,
            embedding_dim=embedding_dim
        )

        per_entity_in_dim = 4 + embedding_dim + 3 + embedding_dim
        self.per_entity_mlp = nn.Sequential(
            nn.Linear(per_entity_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.entities_out_dim = 8

        ############ MLP for scalars

        self.scalar_in_dim = 35 

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.scalars_out_dim = 32

        ############ FINAL LINEAR LAYER

        self.final_agg = nn.Sequential(
            nn.Linear(self.cnn_output_dim + self.entities_out_dim + self.scalars_out_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        1) CNN on map_info + water/lava
        2) MLP on entity info
        3) MLP on scalar data
        4) Concatenate and pass through final aggregator
        """
        # -------------- 1) CNN for map_info --------------
        map_info = observations["map_info"].long()
        water_map = observations["water_map"].float()
        lava_map  = observations["lava_map"].float()

        B = map_info.shape[0]
        # Reshape from (B, 11*21) to (B, 11, 21)
        map_info = map_info.view(B, self.map_height, self.map_width)
        water_map = water_map.view(B, self.map_height, self.map_width)
        lava_map  = lava_map.view(B, self.map_height, self.map_width)

        # Tile embedding => shape (B, 11, 21, embedding_dim)
        map_emb = self.embedding_floor_type(map_info)
        # CNN expects (B, C, H, W), so permute => (B, embedding_dim, 11, 21)
        map_emb = map_emb.permute(0, 3, 1, 2)

        # Add water_map / lava_map as additional channels => each is (B, 1, 11, 21)
        water_map = water_map.unsqueeze(1)
        lava_map  = lava_map.unsqueeze(1)

        # Concat => shape (B, embedding_dim + 2, 11, 21)
        cnn_in = torch.cat([map_emb, water_map, lava_map], dim=1)

        x = self.cnn(cnn_in)          # => (B, flattened_dim)
        cnn_out = self.cnn_linear(x)  # => (B, 128)

        # -------------- 2) Entities --------------
        # We assume you have one big batch dimension B, plus a 'num_entities' dimension
        # (or shape [B, N, ...]) for each field below
        dx = observations["entities_dx"].float()  # shape: (batch_size, 75)
        dy = observations["entities_dy"].float()  # ...
        vx = observations["entities_vx"].float()
        vy = observations["entities_vy"].float()

        type_id   = observations["entities_type_id"].long()   # shape: (batch_size, 75)
        flip_id   = observations["entities_flip"].long()
        holding_id = observations["entities_holding"].long()

        B = dx.shape[0]
        N = dx.shape[1]

        dx_ = dx.unsqueeze(-1)
        dy_ = dy.unsqueeze(-1)
        vx_ = vx.unsqueeze(-1)
        vy_ = vy.unsqueeze(-1)

        type_emb = self.embedding_type_id(type_id)
        flip_oh  = F.one_hot(flip_id, num_classes=3).float()
        hold_emb = self.embedding_type_id(holding_id)

        cat_entities = torch.cat([
            dx_, dy_, vx_, vy_,
            type_emb,
            flip_oh,
            hold_emb
        ], dim=-1)


        # Pass each entity’s vector into the MLP => shape (B, N, 8)
        per_entity_out = self.per_entity_mlp(cat_entities)

        # Sum (or mean) over entities => shape (B, 8)
        entity_agg = per_entity_out.sum(dim=1)
        # or: entity_agg = per_entity_out.mean(dim=1)

        # -------------- 3) Scalar MLP --------------
        health = observations["health"].float()
        bombs = observations["bombs"].float()
        ropes = observations["ropes"].float()
        money = observations["money"].float()
        vel_x = observations["vel_x"].float()
        vel_y = observations["vel_y"].float()
        # face_left = observations["face_left"].float()
        x_rest = observations["x_rest"].float()
        y_rest = observations["y_rest"].float()
        # theme = observations["theme"].float()
        # level = observations["level"].float()
        time_ = observations["time"].float()
        top_holding = observations["holding"].float()
        back = observations["back"].float()
        
        theme = observations["theme"].float()
        theme = theme.squeeze(1)

        level = observations["level"].float()
        level = level.squeeze(1)

        face_left   = observations["face_left"].float()
        # face_left = face_left.argmax(dim=-1)
        face_left = face_left.squeeze(1)

        # for ob in observations:
        #     print(ob, observations[ob].shape)
        # print("----------------")
        # print("health", health.shape)
        # print("bombs", bombs.shape)
        # print("ropes", ropes.shape)
        # print("money", money.shape)
        # print("vel_x", vel_x.shape)
        # print("vel_y", vel_y.shape)
        # print("x_rest", x_rest.shape)
        # print("y_rest", y_rest.shape)
        # print("theme", theme.shape)
        # print("level", level.shape)
        # print("time_", time_.shape)
        # print("top_holding", top_holding.shape)
        # print("back", back.shape)
        # print("face_left", face_left.shape)

        # shape: (B, 14)
        scalar_cat = torch.cat([
            health, bombs, ropes, money, vel_x, vel_y, face_left,
            x_rest, y_rest, theme, level, time_, 
            top_holding, back
        ], dim=1)

        # MLP => (B, 32)
        scalars_out = self.scalar_mlp(scalar_cat)

        # -------------- 4) Combine & Final --------------
        combined = torch.cat([cnn_out, entity_agg, scalars_out], dim=1)
        features = self.final_agg(combined)

        return features


########### INTRINSIC REWARD SYSTEM ###########

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from rllte.xplore.reward import E3B
import torch as th


class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions, 
                         rewards=rewards, terminateds=dones, 
                         truncateds=dones, next_observations=new_obs),
            sync=True)
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #



def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6)
        env = Monitor(env)
        # env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 5000
    # TIMESTEPS = 3000
    TIMESTEPS = 10

    num_envs = 2
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS,
        save_path="models/",
        name_prefix="ppo_spelunky",
        # save_replay_buffer=True,
        save_vecnormalize=True,
    )

    irs = E3B(env, device='cpu')

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        # tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
        policy_kwargs=dict(
            features_extractor_class=SpelunkyFeaturesExtractor,
            features_extractor_kwargs={},
        )
    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            # tb_log_name="14-04-25",
            callback=[
                checkpoint_callback,
                RLeXploreWithOnPolicyRL(irs)
            ],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()