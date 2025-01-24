from pathlib import Path

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

MODELS_ROOT_DIR = Path(__file__).parent.parent.absolute().joinpath('resources').joinpath('models')

YOLO8_MODEL_NAME = 'yolov8n.pt'
YOLO8_MODEL_DIR = MODELS_ROOT_DIR.joinpath(YOLO8_MODEL_NAME.split('.')[0])

ZOO_MODEL_NAME = 'person-detection-0202'
ZOO_MODEL_PRECISION = 'FP16'
ZOO_MODEL_DIR = MODELS_ROOT_DIR.joinpath(ZOO_MODEL_NAME).joinpath(ZOO_MODEL_PRECISION)
ZOO_MODEL_DOWNLOAD_URL = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{ZOO_MODEL_NAME}/{ZOO_MODEL_PRECISION}/{ZOO_MODEL_NAME}.xml"

