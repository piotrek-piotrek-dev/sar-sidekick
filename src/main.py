from pathlib import Path

import PIL
import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO
from pprint import pprint
from deep_sort_realtime.deepsort_tracker import DeepSort

from color_detection_5 import draw_bbox_on_image_2, color_detection
from helpers.Detection import Detection
from helpers.LoggerConfig import get_logger
from helpers.constants import DEEP_SORT_MAX_AGE, YOLO8_MODEL_PATH, CONFIDENCE_THRESHOLD, INPUT_FRAME_FILE_PATH, PERSON_CLASS_ID


log = get_logger(__name__)

def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def detect(mmodel, frame: Path | str | PIL.Image.Image | np.ndarray, show_intermediate_frame=False) -> list:
    """
    :param mmodel: YOLO model to run inference. YOLO constructor
    :param frame: path to frame on which we need to run detection
    :param show_intermediate_frame: flag to steer if you want to see all detections as bboxes
    :return: list of tuples containing:
    - bounding box in format of a list: [x, y, w, h] of each bbox
    - confidence (in %) of detection in a given bbox
    - detected class id
    """
    log.info("entering detection module")
    # run the YOLO model on the frame
    detections = mmodel(frame)[0]
    log.info("finished detections, found %s objects in %s", len(detections.boxes.data.tolist))
    if show_intermediate_frame:
        img = Image.fromarray(detections.plot()[:, :, ::-1])
        img.show()
    results = []
    # loop over the detections
    for data in detections.boxes.data.tolist():
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
    return results


if __name__ == '__main__':
    log.info("entering main method")
    log.info("loading model")
    # load the pre-trained YOLOv8n model
    model = YOLO(YOLO8_MODEL_PATH)
    #tracker = DeepSort(max_age=DEEP_SORT_MAX_AGE)
    log.info("loaded model")
    """"# initialize the video capture object
    video_cap = cv2.VideoCapture(INPUT_VIDEO_FILE)
    # initialize the video writer object
    writer = create_video_writer(video_cap, OUTPUT_FILE_NAME)

    while True:
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        if not ret:
            break

        results = detect(mmodel, frame)
        pprint(results)
    """

    original_image = cv2.imread(INPUT_FRAME_FILE_PATH)
    detections = detect(model, INPUT_FRAME_FILE_PATH)

    #bounding_boxes = [(det[0][0], det[0][1], det[0][2], det[0][3]) for det in detections]
    person_bounding_boxes: list[Detection]
    person_bounding_boxes = [Detection(INPUT_FRAME_FILE_PATH,
                                       x=det[0][0], y=det[0][1], width=det[0][2], height=det[0][3],
                                       confidence=det[1])
                             for det in detections
                             if det[2] == PERSON_CLASS_ID]

    pprint(person_bounding_boxes)

    image = original_image.copy()
    # processed_frame: ProcessedFrame = ProcessedFrame(image, INPUT_FRAME_FILE_PATH)
    for bbox in person_bounding_boxes:
        image = draw_bbox_on_image_2(image, bbox)
        # uncomment below 2 lines if you want to see all the intermediate bboxes on the image
        # intermediate_img = Image.fromarray(image)
        # intermediate_img.show()

    # yet another intermediate checkpoint for visualization
    processed_image = Image.fromarray(image)
    processed_image.show()

    processed_frame = \
        {
            'name': INPUT_FRAME_FILE_PATH,
            'bboxes': [bbox.get_bouding_box() for bbox in person_bounding_boxes],
            'image': original_image,
        }

    color_detection([processed_frame])
