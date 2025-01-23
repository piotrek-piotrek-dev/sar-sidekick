"""
this script is for setting up your environment to implement the DeepSort algorithm for person detection using open Vino
credits and sources from: https://docs.openvino.ai/2024/notebooks/person-tracking-with-output.html#select-inference-device
"""

import subprocess
import sys
from pathlib import Path

PACKAGES_TO_INSTALL = [
    "openvino>=2024.0.0",
    "opencv-python",
    "requests",
    "scipy",
    "tqdm",
    "matplotlib>=3.4",
]

NOTEBOOK_WEB_LOCATION = r"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"


def is_package_installed(packageName: str) -> bool:
    if packageName in sys.modules:
        print(f"package {packageName} already installed\n")
        return True
    else:
        print(f"package {packageName} not installed, attempting to install\n")
        return False

def install_notebook() -> bool:
    if not Path("./notebook_utils.py").exists():
        #Fetch `notebook_utils` module
        import requests
        r = requests.get(
            url=NOTEBOOK_WEB_LOCATION,
        )
        if r.status_code is not 200: return False

        with open("notebook_utils.py", "w") as f: f.write(r.text)
        r.close()
        return True

def install_dependencies() -> bool:
    for package in PACKAGES_TO_INSTALL:
        if not is_package_installed(package):
            if not install_package(package): return False
    if not install_notebook(): return False
    return True

def install_package(packageName: str) -> bool:
    try:
        print(f"Installing {packageName}...\n")
        ret = subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
        if ret == 0:
            print(f"package {packageName} is installed\n")
            return True
    except subprocess.CalledProcessError as e:
        print(f"something went wrong with installing {packageName}\n"
              f"check output: {e.output}")
        return False


if __name__ == '__main__':
    if not install_dependencies():
        sys.exit(1)
