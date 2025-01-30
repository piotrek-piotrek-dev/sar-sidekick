import numpy as np
from torch.nn.functional import selu_

from helpers.Detection import Detection


class ProcessedFrame:
    def __init__(self, frame: np.ndarray, name: str):
        self.image: np.ndarray = frame
        self.name:str = name
        self.bboxes: [(int, int, int, int)] = None

    def append(self, detections: Detection):
        self.bboxes.append(detections)
