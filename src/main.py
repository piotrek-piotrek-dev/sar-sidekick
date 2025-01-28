

import datetime
from pprint import pprint

from PIL import Image
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.constants import INPUT_VIDEO_FILE_NAME, OUTPUT_FILE_NAME, MODEL_NAME, DEEP_SORT_MAX_AGE, INPUT_VIDEO_FILE, \
    YOLO8_MODEL_PATH, CONFIDENCE_THRESHOLD, INPUT_FRAME_FILE_PATH


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

def detect(mmodel, frame):
    # run the YOLO model on the frame
    detections = mmodel(frame)[0]
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

    # load the pre-trained YOLOv8n model
    model = YOLO(YOLO8_MODEL_PATH)
    tracker = DeepSort(max_age=DEEP_SORT_MAX_AGE)

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

    # detections = model(INPUT_FRAME_FILE_PATH)
    detections = detect(model, INPUT_FRAME_FILE_PATH)

    bounding_boxes = [(det[0][0], det[0][1], det[0][2], det[0][3]) for det in detections]

    pprint(bounding_boxes)


