import cv2
import numpy as np
from pathlib import Path



class Detection:
    #def __init__(self, image: Path | str | PIL.Image.Image | np.ndarray, x: int, y: int, width: int, height: int):
    def __init__(self, image: str | Path,
                 x: int,
                 y: int,
                 width: int,
                 height: int,
                 confidence: float):
        self.image: str = image
        # if isinstance(image, Path):
        #     self.image = cv2.imread(str(image))
        # elif isinstance(image, np.ndarray):
        #     #image = Image.fromarray(image)
        #     self.image = image
        # elif isinstance(image, PIL.Image.Image):
        #     # TODO: image = image.resize((w, h))
        #     pass
        # elif isinstance(image, str) or isinstance(image, Path):
        #     self.image = cv2.imread(image)

        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.confidence: float = confidence

    def get_bouding_box(self) -> [int]:
        return [self.x, self.y, self.width, self.height]

    def get_original_image(self) -> np.ndarray:
        if isinstance(self.image, Path):
            return cv2.imread(str(self.image))
        else:
            return cv2.imread(self.image)

    def get_original_image_path(self) -> Path:
        return Path(self.image)

    def get_confidence(self) -> float:
        return self.confidence
