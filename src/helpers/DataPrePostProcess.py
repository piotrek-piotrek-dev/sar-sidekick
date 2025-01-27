# this is a helper file with methods for data pre and post process
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Preprocess a single image

    Parameters
    ----------
    frame: input frame
    height: height of model input data
    width: width of model input data
    """
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image

def batch_preprocess(img_crops: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Preprocess batched images

    Parameters
    ----------
    img_crops: batched input images
    height: height of model input data
    width: width of model input data
    """
    img_batch = np.concatenate([preprocess_frame(img, height, width) for img in img_crops], axis=0)
    return img_batch

def process_results(h, w, results, threshold=0.5) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    postprocess detection results

    Parameters
    ----------
    h, w: original height and width of input image
    results: raw detection network output
    threshold: threshold for low confidence filtering
    """
    # The 'results' variable is a [1, 1, N, 7] tensor.
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        # Filter detected objects.
        if score > threshold:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                [
                    (xmin + xmax) / 2 * w,
                    (ymin + ymax) / 2 * h,
                    (xmax - xmin) * w,
                    (ymax - ymin) * h,
                    ]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)]
    return tuple(color)

def draw_boxes_on_frame(img: np.ndarray, bbox, identities=None) -> np.ndarray:
    """
    Draw bounding box in original image

    Parameters
    ----------
    img: original image
    bbox: coordinate of bounding box
    identities: identities IDs
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            [255, 255, 255],
            2,
        )
    return img

def cosin_metric(x1, x2):
    """
    Calculate the consin distance of two vector

    Parameters
    ----------
    x1, x2: input vectors
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
