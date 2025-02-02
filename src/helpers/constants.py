import logging
from datetime import datetime
from pathlib import Path

PROJECT_NAME = 'sar-sidekick'
PROJECT_ROOT_DIR = [p for p in Path(__file__).parents if p.parts[-1]==PROJECT_NAME][0]
RESOURCES_ROOT_DIR = PROJECT_ROOT_DIR.joinpath('resources')
MODELS_ROOT_DIR = RESOURCES_ROOT_DIR.joinpath('models')
LOGS_ROOT_DIR = RESOURCES_ROOT_DIR.joinpath('logs')
LOG_LEVEL = logging.INFO
PRESENTATION_MODE = True
BETTER_GRAB_YOURSELF_A_COFFE = False

def get_date_time_as_str() -> str:
    time = (str(datetime.now())
            .replace(' ', '_')
            .replace(':', '-')
            .replace('.', '-'))
    return time

LOG_TO_STD_OUT = True
LOG_FILE_PATH = LOGS_ROOT_DIR.joinpath(f'log_{get_date_time_as_str()}.log')

CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
PERSON_CLASS_ID = 0

YOLO8_MODEL_NAME = 'yolov8n.pt'
YOLO8_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO8_MODEL_NAME.split('.')[0])
YOLO8_MODEL_PATH = YOLO8_MODEL_DIR.joinpath(YOLO8_MODEL_NAME)

INPUT_VIDEO_FILE_NAME = '2.mp4'
INPUT_VIDEO_FILE = Path(RESOURCES_ROOT_DIR.joinpath('video').joinpath(INPUT_VIDEO_FILE_NAME))
OUTPUT_FILE_NAME = 'output.mp4'

INPUT_FRAME_FILE_NAME = '1_test.jpeg'
INPUT_FRAME_FILE_PATH = Path(RESOURCES_ROOT_DIR.joinpath('img').joinpath(INPUT_FRAME_FILE_NAME))

DEEP_SORT_MAX_AGE = 50

YOLO11_MODEL_NAME = 'YOLO11s-seg.pt'
YOLO11_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO11_MODEL_NAME.split('.')[0])
YOLO11_MODEL_PATH = YOLO11_MODEL_DIR.joinpath(YOLO11_MODEL_NAME)

DETECTION_MODEL_NAME = 'yolo11n.pt'
DETECTION_MODEL_DIR = MODELS_ROOT_DIR.joinpath(DETECTION_MODEL_NAME.split('.')[0])
DETECTION_MODEL_PATH = YOLO11_MODEL_DIR.joinpath(DETECTION_MODEL_NAME)
