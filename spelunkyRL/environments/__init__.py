import os
import importlib

# Get the current directory (where this __init__.py lives)
current_dir = os.path.dirname(__file__)

# Loop through all .py files in the directory
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # remove .py extension
        importlib.import_module(f".{module_name}", package=__name__)
