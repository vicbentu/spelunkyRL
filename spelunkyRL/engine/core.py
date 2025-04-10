import os, socket, subprocess, json, atexit
from datetime import datetime

import gymnasium as gym

from . import config

class SpelunkyRLEngine(gym.Env):

    ############## GYM interface ##############

    # Frameskipping, 
    def __init__(self, frames_per_step=4, speedup=True):
        super().__init__()
        
        self.frames_per_step = frames_per_step
        self.speedup = speedup

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
            "input": action.tolist(),
            "frames": self.frames_per_step,
        })
        
        gamestate = self._receive_dict()
        done = bool(gamestate["player_info"]["health"] <= 0 or gamestate["screen_info"]["win"] == 1)

        # # PRINT MAP INFO
        # with open(config.log_file, "a") as f:
        #     for row in gamestate["screen_info"]["map_info"]:
        #         formatted_row = " ".join(f"{num:4}" for num in row)  # 4-character width
        #         f.write(f"{formatted_row}\n")
        # # PRINT ENTITIESº
        # from collections import Counter
        # from ..tools.id2name import id2name
        # type_counts = Counter(id2name(entity[4])["name"] for entity in gamestate["entity_info"])
        # with open(config.log_file, "a") as f:
        #     f.write(f"Entities: {type_counts}\n")

        reward = self.reward_function(gamestate, self.last_gamestate)
        self.last_gamestate = gamestate
        observation = self.gamestate_to_observation(gamestate)

        return observation, reward, done, False, {}
    
    action_space = gym.spaces.MultiDiscrete([
        3, # Movement X
        3, # Movement Y
        2, # Jump
        2, # Whip
        2, # Bomb
        2, # Rope
        2, # Run
        2, # Door
    ]) 

    ############ Spelunky  Communicaton ############

    def close(self):
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

        self.server_socket.settimeout(5.0)

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
            "command": "reset",
            "speedup": self.speedup,
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
        
        if "log_file" in config.__dict__:
            with open(config.log_file, "a") as f:
                timestamp = datetime.now().strftime("%H:%M:%S:%f")[:-3]
                f.write(f"-- {timestamp} {str(dict)}\n")
            
        return dict