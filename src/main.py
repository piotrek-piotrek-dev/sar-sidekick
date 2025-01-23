"""
this script implements the DeepSort algorithm for person detection using open Vino
credits and sources from: https://docs.openvino.ai/2024/notebooks/person-tracking-with-output.html#select-inference-device
"""

import sys


if __name__ == '__main__':
    print(r"Starting main script")
    from utils import setup
    if not setup.install_dependencies():
        sys.exit(1)