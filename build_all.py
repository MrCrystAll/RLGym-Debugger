from pathlib import Path
import shutil
import subprocess
import sys

INSTALL_PATHS = [
    "rlgym_debugger/api",
    "rlgym_debugger_human_player/api",
    "rlgym_debugger_human_player/rocket_league",
    "rlgym_debugger_multi_agents/api",
]

for install_path in INSTALL_PATHS:
    install_path = Path(install_path)
    result = subprocess.run(
        [sys.executable, "-m", "build", install_path],
        check=False
    )
    
    if result.returncode != 0:
        raise ValueError(f"The build for the package at path {install_path} did not finish correctly, stopping.")