import os, socket, subprocess, json, atexit
from datetime import datetime

import numpy as np
import gymnasium as gym

from . import config

class SpelunkyEnv(gym.Env):

    ############## GYM interface ##############

    def __init__(self, observation_space=None, reward_function=None, gamestate_to_observation=None):
        super().__init__()
        self.action_space = gym.spaces.MultiDiscrete([
            3, # Movement X
            3, # Movement Y
            2, # Jump
            2, # Whip
            2, # Bomb
            2, # Rope
            2, # Run
            2, # Door
        ])       

        self.observation_space = observation_space
        self.reward_function = reward_function
        self.gamestate_to_observation = gamestate_to_observation

        # Start Spelunky
        self._game_init()


    # TODO: seed, level, items, gold, hp
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._game_reset(seed=seed)
        
        gamestate = self._receive_dict() 
        self.last_gamestate = gamestate
        observation = self.gamestate_to_observation(gamestate)
        return observation, {}


    def step(self, action):
        self._send_dict({
            "command": "step",
            "input": action.tolist()
        })
        
        gamestate = self._receive_dict()
        done = bool(gamestate["health"] <= 0 or gamestate["win"] == 1)

        reward = self.reward_function(gamestate, self.last_gamestate)
        self.last_gamestate = gamestate
        
        observation = self.gamestate_to_observation(gamestate)

        if "log_file" in config.__dict__:
            with open(config.log_file, "a") as f:
                timestamp = datetime.now().strftime("%H:%M:%S:%f")[:-3]
                f.write(f"-- {timestamp} {str(observation)}\n")

        return observation, reward, done, False, {}


    ############ Spelunky  Communicaton ############

    def close(self):
        return
        self._send_dict({
            "command": "close"
        })

    def _game_init(self):

        executable = "playlunky_launcher.exe"
        args = [
            f'-exe_dir={config.spelunky_dir}',
            *(['-console'] if config.console else [])
        ]
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('127.0.0.1', 0))  # bind to any available port
        self.server_socket.listen(1)  # listen for 1 connection
        port = self.server_socket.getsockname()[1]
        os.environ["Spelunky_RL_Port"] = str(port)

        self.game_process = subprocess.Popen(
            [executable] + args,
            cwd=config.playlunky_dir,
            shell=True
        )

        self.server_socket.settimeout(1.0)

        while True:
            try:
                self.server, addr = self.server_socket.accept()
                break
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
        
        atexit.register(self.close)

    def _game_reset(self, seed = None):
        self._send_dict({
            "command": "reset"
        })

    def _send_dict(self, dict):
        json_str = json.dumps(dict) + "\n"
        self.server.sendall(json_str.encode("utf-8"))

    def _receive_dict(self):
        buffer = b""
        while not buffer.endswith(b"\n"):
            data = self.server.recv(1024)
            if not data:
                raise ConnectionError("Disconnected from Spelunky Lua script")
            buffer += data
        json_str = buffer.decode("utf-8").strip()
        dict = json.loads(json_str)
        if "error" in dict:
            raise RuntimeError(dict["error"])
        return dict


# if __name__ == '__main__':
#     env = SpelunkyEnv()
#     env.reset()

#     startTime = datetime.now()

#     counter = 0
#     while True:
#         counter += 1
#         if counter % 60 == 0:
#             print(f"{counter/60} seconds have passed, FPS: {counter/(datetime.now()-startTime).total_seconds()}")
#         observation, reward, _, done, _ = env.step(env.action_space.sample())
#         if done:
#             env.reset()