import os, socket, subprocess, json, atexit, psutil, random
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional


import gymnasium as gym

import mss
import numpy as np
import win32gui, win32con, win32process, win32api, ctypes
import win32ui
from PIL import Image
from ..tools.frame_grabber import FrameGrabber

from . import config
from spelunkyRL.tools.window_management import get_hwnd_for_pid, force_foreground_window, ensure_window_visible

class SpelunkyRLEngine(gym.Env):

    ############## GYM interface ##############

    action_space: gym.spaces.MultiDiscrete = gym.spaces.MultiDiscrete([
        3, # Movement X
        3, # Movement Y
        2, # Jump

        # 2, # Whip
        # 2, # Bomb
        # 2, # Rope
        # 2, # Run
        # 2, # Door
    ]) 

    observation_space: gym.spaces.Dict

    def __init__(self, frames_per_step: int = 6, speedup: bool = True, reset_options: dict = {}, render_enabled: bool = False) -> None:
        super().__init__()

        self.frames_per_step = frames_per_step
        self.speedup = speedup
        self.reset_options = reset_options
        self.render_enabled = render_enabled
        self.render_mode = 'rgb_array'

        # Start Spelunky
        self._game_init()


    # TODO: seed, level, items, gold, hp
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Dict, Dict[str, Any]]:
        
        super().reset(seed=seed)
        if options is None: # TODO: Substitute options individually
            options = self.reset_options
        self._game_reset(seed=seed, **options)
        
        gamestate = self._receive_dict() 
        self.last_gamestate = gamestate
        observation = self.gamestate_to_observation(gamestate)
        return observation, {}


    def step(
        self, action: Any
    ) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:

        self._send_dict({
            "command": "step",
            "input": action.tolist() + [0,0,0,1,0],
            # "input": [1,1,0,0] + action.tolist() + [0,0,0],
            # "input": action.tolist(),
            "frames": self.frames_per_step,
        })
        
        gamestate = self._receive_dict()
        done = bool(gamestate["player_info"]["health"] <= 0 or gamestate["screen_info"]["win"] == 1)

        # PRINT MAP INFO
        # with open(r"log.txt", "a") as f:
        #     # f.write(str(gamestate["screen_info"]["map_info"]) + "\n")
        #     f.write("----------------------------------\n")
        #     for row in gamestate["screen_info"]["map_info"]:
        #         formatted_row = " ".join(f"{cell[0]:4}" for cell in row)
        #         f.write(f"{formatted_row}\n")
        # PRINT ENTITIESº
        # from collections import Counter
        # from ..tools.id2name import id2name
        # type_counts = Counter(id2name(entity[4])["name"] for entity in gamestate["entity_info"])
        # with open(config.log_file, "a") as f:
        #     f.write(f"Entities: {type_counts}\n")

        reward, done, truncated = self.reward_function(gamestate, self.last_gamestate, action, done)
        self.last_gamestate = gamestate
        observation = self.gamestate_to_observation(gamestate)

        return observation, reward, done, truncated, {}
    
    

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

        self.launcher_process = subprocess.Popen(
            [executable] + args,
            cwd=config.playlunky_dir,
            shell=True
        )
        self.game_process = None
        self.server_socket.settimeout(0.05)



        while True:
            try:
                self.server, addr = self.server_socket.accept()
                break
            except socket.timeout:
                pass

            if self.game_process is not None and self.game_process.is_running():
                pass
            else:
                if self.launcher_process.poll() is None:
                    parent = psutil.Process(self.launcher_process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        if child.name().startswith("Spel2"):
                            self.game_process = child

                else:
                    self.launcher_process = subprocess.Popen(
                        [executable] + args,
                        cwd=config.playlunky_dir,
                        shell=True
                    )

        atexit.register(self.close)
        self.hwnd = get_hwnd_for_pid(self.game_process.pid)
        if self.render_enabled:
            self.grabber = FrameGrabber(self.hwnd)

    def _game_reset(self, seed = None, ent_types_to_destroy = []) -> None:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
        self._send_dict({
            "command": "reset",
            "speedup": self.speedup,
            "seed": seed,
            "ent_types_to_destroy": ent_types_to_destroy,
        })

    def _send_dict(self, payload: Dict[str, Any]) -> None:
        json_str = json.dumps(payload) + "\n"
        self.server.sendall(json_str.encode("utf-8"))
    
    def _receive_dict(self) -> Dict[str, Any]:
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

    
    ############ Render ############ 
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        if self.render_enabled is False:
            raise RuntimeError("Use render_enabled=True on init to be able to record replays")
        
        return self.grabber.frame.copy()