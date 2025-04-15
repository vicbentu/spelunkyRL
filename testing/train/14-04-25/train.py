# Parallel training
# Added embedings for entities and map info
# Added different aggregation methods for entities, map info and scalars
# Discretes as boxes because got auto-converted OneHotVectors

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

class SpelunkyFeaturesExtractor(BaseFeaturesExtractor):

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

        self.scalar_in_dim = 34 

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
        map_info = map_info.view(B, self.map_height, self.map_width)
        water_map = water_map.view(B, self.map_height, self.map_width)
        lava_map  = lava_map.view(B, self.map_height, self.map_width)

        map_emb = self.embedding_floor_type(map_info)
        map_emb = map_emb.permute(0, 3, 1, 2)

        water_map = water_map.unsqueeze(1)
        lava_map  = lava_map.unsqueeze(1)

        cnn_in = torch.cat([map_emb, water_map, lava_map], dim=1)

        x = self.cnn(cnn_in)
        cnn_out = self.cnn_linear(x)

        # -------------- 2) Entities --------------
        dx = observations["entities_dx"].float()
        dy = observations["entities_dy"].float()
        vx = observations["entities_vx"].float()
        vy = observations["entities_vy"].float()

        type_id   = observations["entities_type_id"].long()
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


        per_entity_out = self.per_entity_mlp(cat_entities)

        entity_agg = per_entity_out.sum(dim=1)

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
        face_left = face_left.argmax(dim=-1)

        for ob in observations:
            print(ob, observations[ob].shape)

        scalar_cat = torch.cat([
            health, bombs, ropes, money, vel_x, vel_y, face_left,
            x_rest, y_rest, theme, level, time_, 
            top_holding, back
        ], dim=1)

        scalars_out = self.scalar_mlp(scalar_cat)

        # -------------- 4) Combine & Final --------------
        combined = torch.cat([cnn_out, entity_agg, scalars_out], dim=1)
        features = self.final_agg(combined)

        return features



def make_env():
    def _init():
        env = SpelunkyEnv(frames_per_step=6)
        env = Monitor(env)
        # env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":

    TOTAL_LOOPS = 5000
    TIMESTEPS = 3000
    # TIMESTEPS = 300

    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    checkpoint_callback = CheckpointCallback(
        save_freq=TIMESTEPS,
        save_path="models/",
        name_prefix="ppo_spelunky",
        # save_replay_buffer=True,
        save_vecnormalize=True,
    )


    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        device="cuda",
        n_steps=TIMESTEPS,
        tensorboard_log=r"C:\TFG\Project\SpelunkyRL\Spelunky 2\Mods\Packs\spelunkyRL\tensorboardlogs",
        policy_kwargs=dict(
            features_extractor_class=SpelunkyFeaturesExtractor,
            features_extractor_kwargs={},
        )
    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS * TOTAL_LOOPS,
            tb_log_name="14-04-25",
            callback=[
                checkpoint_callback,
            ],
            reset_num_timesteps=False
        )

    except Exception as e:
        from datetime import datetime
        print(f"Current time: {datetime.now()}")
        print(f"Error: {e}")
        env.close()