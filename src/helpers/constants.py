from pathlib import Path

from helpers.LoggerConfig import get_date_time_as_str

PROJECT_NAME = 'sar-sidekick'

LOG_TO_STD_OUT = True

CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
PERSON_CLASS_ID = 0

PROJECT_ROOT_DIR = [p for p in Path(__file__).parents if p.parts[-1]==PROJECT_NAME][0]
RESOURCES_ROOT_DIR = PROJECT_ROOT_DIR.joinpath('resources')
MODELS_ROOT_DIR = RESOURCES_ROOT_DIR.joinpath('models')

LOG_FILE_PATH = PROJECT_ROOT_DIR.joinpath(f'log_{get_date_time_as_str()}.log')

YOLO8_MODEL_NAME = 'yolov8n.pt'
YOLO8_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO8_MODEL_NAME.split('.')[0])
YOLO8_MODEL_PATH = YOLO8_MODEL_DIR.joinpath(YOLO8_MODEL_NAME)

INPUT_VIDEO_FILE_NAME = '2.mp4'
INPUT_VIDEO_FILE = Path(RESOURCES_ROOT_DIR.joinpath('video').joinpath(INPUT_VIDEO_FILE_NAME))
OUTPUT_FILE_NAME = 'output.mp4'

INPUT_FRAME_FILE_NAME = '3.jpg'
INPUT_FRAME_FILE_PATH = Path(RESOURCES_ROOT_DIR.joinpath('img').joinpath(INPUT_FRAME_FILE_NAME))

DEEP_SORT_MAX_AGE = 50

YOLO11_MODEL_NAME = 'YOLO11n-seg.pt'
YOLO11_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO11_MODEL_NAME.split('.')[0])
YOLO11_MODEL_PATH = YOLO11_MODEL_DIR.joinpath(YOLO11_MODEL_NAME)
