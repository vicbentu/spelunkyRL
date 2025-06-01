SpelunkyRL is a Reinforcement Learning environment, designed to provide a standard RL interface to Spelunky 2. It comes with predefined tasks but also a high degree of customability, to allow for a wide variety of experimentation.

This project is built using 
[Playlunky](https://github.com/spelunky-fyi/Playlunky),
[modlunky2](https://github.com/spelunky-fyi/modlunky2) and
[overlunky](https://github.com/spelunky-fyi/overlunky).
Some information will reference these, specially the latter one, when talking about identifiers and enumerations.

# 🚀 Getting Started

This project uses [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), which provides a standardized interface for RL environments. If you're familiar with it, using SpelunkyRL should be extremely simple.

## 🛠️ Installation
To use this project, you require an installation of Spelunky 2 on your computer. It also requires `python>=3.8` and `Windows 11` (probably works with older version, but some features like "render" might give problems).
The first step you should do is install
[modlunky2](https://github.com/spelunky-fyi/modlunky2), following the instructions. As already explained there, it's not recommended using modding tools with your actual Steam installation.

Once you've completed the first step, clone this repo into `..\Spelunky 2\Mods\Packs\`, being `..` your Spelunky 2 installation path.
Then, create a virtual environment where you want to work (it can be that same folder if you prefer). Finally, install the library using
```bash
# If working in the same directory as the cloned repo
pip install .

# If working from a different location, provide the full path to the repo
pip install "C:\Path\To\Your\Spelunky 2\Mods\Packs\spelunkyRL"
```
After that, you're all set to start using SpelunkyRL!

## 💡 Basic Usage

To use any of the default environments, simply import it on your script this way. You should have the path to your Spelunky 2 game and to your playlunky installation. Both can be obtained executing modlunky and going to the settings window. The first one will appear right there, and for the second one you should click "User Directories" -> "Data". That will open a file explorer, enter the "playlunky" dir and there will be all the playlunky vesions. Copy the path to any of them.

![alt text](doc/modlunky2config.png)

```python
from spelunkyRL.environments.dummy_environment import SpelunkyEnv

env = SpelunkyEnv(
    spelunky_dir= # path to the dir where Spel2.exe is
    playlunky_dir= # path to the dir where playlunky_launcher.exe is
)

env.reset()
env.step(...)
```

## 📘 Environment Interface & Method Documentation

Here are the methods included in the environment and the parameters each one accepts:


## 🧪 Build your Custom Environment