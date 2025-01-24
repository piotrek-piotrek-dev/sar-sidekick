"""
this script implements the DeepSort algorithm for person detection using open Vino
credits and sources from: https://docs.openvino.ai/2024/notebooks/person-tracking-with-output.html#select-inference-device
and https://github.com/python-dontrepeatyourself/Real-Time-Object-Tracking-with-DeepSORT-and-YOLOv8/
"""

import sys
import cv2
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
import OpenVINO_model_wrapper

from IPython import display
from pathlib import Path

from src.Constants import ZOO_MODEL_NAME
from src.helpers.LoggingConfig import createLogger


log = createLogger(__name__)

if __name__ == '__main__':
    log.debug('Entering main module.')
    """
    USE THIS ONLY ONCE AT STARTUP
    from helpers import setup
    if not setup.install_dependencies():
        sys.exit(1)
    if not setup.import_modules():
        sys.exit(1)
    """
    # download the model, assume it's in ZOO_MODEL_DIR

    model = OpenVINO_model_wrapper.OpenVINO_model_wrapper(model_path=ZOO_MODEL_NAME)

