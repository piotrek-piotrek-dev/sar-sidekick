from pathlib import Path

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

MODELS_ROOT_DIR = Path(__file__).parent.parent.absolute().joinpath('resources').joinpath('models')
RESOURCES_ROOT_DIR = Path(__file__).parent.parent.absolute().joinpath('resources')

YOLO8_MODEL_NAME = 'yolov8n.pt'
YOLO8_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO8_MODEL_NAME.split('.')[0])
YOLO8_MODEL_PATH = YOLO8_MODEL_DIR.joinpath(YOLO8_MODEL_NAME).joinpath(YOLO8_MODEL_NAME)

INPUT_VIDEO_FILE_NAME = '2.mp4'
INPUT_VIDEO_FILE = Path(RESOURCES_ROOT_DIR.joinpath('video').joinpath(INPUT_VIDEO_FILE_NAME))
OUTPUT_FILE_NAME = 'output.mp4'

INPUT_FRAME_FILE_NAME = '3.jpg'
INPUT_FRAME_FILE_PATH = Path(RESOURCES_ROOT_DIR.joinpath('img').joinpath(INPUT_FRAME_FILE_NAME))

MODEL_NAME = 'yolov8n.pt'
DEEP_SORT_MAX_AGE = 50
