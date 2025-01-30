from pathlib import Path

from jupyter_core.version import parts
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import multiprocessing
import threading
import queue
import matplotlib.pyplot as plt

from helpers.Detection import Detection
from helpers.ProcessedFrame import ProcessedFrame
from helpers.constants import YOLO11_MODEL_PATH

# Dopacowanie do kolorow przedziałów, w HSV (wg neta OpenCV lepiej na tym dziala niz na klasycznym RGB)
COLOR_RANGES = {
    "czerwony": [
        {"lower": np.array([0, 150, 100], dtype=np.uint8), "upper": np.array([10, 255, 255], dtype=np.uint8)},
        # Standardowy czerwony
        {"lower": np.array([170, 150, 100], dtype=np.uint8), "upper": np.array([180, 255, 255], dtype=np.uint8)},
        {"lower": np.array([0, 150, 50], dtype=np.uint8), "upper": np.array([10, 255, 150], dtype=np.uint8)},
        # Ciemny czerwony
        {"lower": np.array([0, 150, 200], dtype=np.uint8), "upper": np.array([10, 255, 255], dtype=np.uint8)},
        # Jasny czerwony
    ],
    "zielony": [
        {"lower": np.array([35, 70, 70], dtype=np.uint8), "upper": np.array([85, 255, 255], dtype=np.uint8)},
        # Standardowy zielony
        {"lower": np.array([35, 70, 50], dtype=np.uint8), "upper": np.array([85, 255, 150], dtype=np.uint8)},
        # Ciemny zielony
        {"lower": np.array([35, 70, 200], dtype=np.uint8), "upper": np.array([85, 255, 255], dtype=np.uint8)},
        # Jasny zielony
    ],
    "niebieski": [
        {"lower": np.array([100, 150, 70], dtype=np.uint8), "upper": np.array([130, 255, 255], dtype=np.uint8)},
        # Standardowy niebieski
        {"lower": np.array([100, 150, 50], dtype=np.uint8), "upper": np.array([130, 255, 150], dtype=np.uint8)},
        # Ciemny niebieski
        {"lower": np.array([100, 150, 200], dtype=np.uint8), "upper": np.array([130, 255, 255], dtype=np.uint8)},
        # Jasny niebieski
    ],
    "żółty": [
        {"lower": np.array([20, 150, 150], dtype=np.uint8), "upper": np.array([30, 255, 255], dtype=np.uint8)},
        # Standardowy żółty
        {"lower": np.array([20, 150, 50], dtype=np.uint8), "upper": np.array([30, 255, 150], dtype=np.uint8)},
        # Ciemny żółty
        {"lower": np.array([25, 180, 240], dtype=np.uint8), "upper": np.array([30, 255, 255], dtype=np.uint8)},
        # Neonowy żółty
        {"lower": np.array([25, 50, 200], dtype=np.uint8), "upper": np.array([30, 100, 255], dtype=np.uint8)},
        # Jasny pastelowy
        {"lower": np.array([26, 50, 170], dtype=np.uint8), "upper": np.array([30, 140, 255], dtype=np.uint8)},
        # Kanarkowy żółty i jego warianty (#f9ec6f, #fce955, itp.)
    ],
    "pomarańczowy": [
        {"lower": np.array([10, 150, 150], dtype=np.uint8), "upper": np.array([20, 255, 255], dtype=np.uint8)},
        # Standardowy pomarańczowy
        {"lower": np.array([10, 150, 50], dtype=np.uint8), "upper": np.array([20, 255, 150], dtype=np.uint8)},
        # Ciemny pomarańczowy
        {"lower": np.array([10, 150, 200], dtype=np.uint8), "upper": np.array([20, 255, 255], dtype=np.uint8)},
        # Jasny pomarańczowy
    ],
    "fioletowy": [
        {"lower": np.array([130, 50, 50], dtype=np.uint8), "upper": np.array([155, 255, 255], dtype=np.uint8)},
        # Standardowy fioletowy
        {"lower": np.array([130, 50, 30], dtype=np.uint8), "upper": np.array([155, 255, 120], dtype=np.uint8)},
        # Ciemny fioletowy
        {"lower": np.array([130, 50, 200], dtype=np.uint8), "upper": np.array([155, 255, 255], dtype=np.uint8)},
        # Jasny fioletowy
    ],
    "różowy": [
        {"lower": np.array([160, 50, 70], dtype=np.uint8), "upper": np.array([175, 255, 255], dtype=np.uint8)},
        # Standardowy różowy
        {"lower": np.array([160, 50, 50], dtype=np.uint8), "upper": np.array([175, 255, 150], dtype=np.uint8)},
        # Ciemny różowy
        {"lower": np.array([160, 50, 200], dtype=np.uint8), "upper": np.array([175, 255, 255], dtype=np.uint8)},
        # Jasny różowy
    ],
    "brązowy": [
        {"lower": np.array([10, 50, 20], dtype=np.uint8), "upper": np.array([30, 200, 150], dtype=np.uint8)},
        # Standardowy brązowy
        {"lower": np.array([10, 50, 10], dtype=np.uint8), "upper": np.array([30, 150, 100], dtype=np.uint8)},
        # Ciemny brązowy
        {"lower": np.array([10, 50, 160], dtype=np.uint8), "upper": np.array([30, 255, 255], dtype=np.uint8)},
        # Jasny brązowy
    ],
    "biały": [
        {"lower": np.array([0, 0, 220], dtype=np.uint8), "upper": np.array([180, 30, 255], dtype=np.uint8)},
        # Standardowy biały
        {"lower": np.array([0, 0, 200], dtype=np.uint8), "upper": np.array([180, 30, 220], dtype=np.uint8)},
        # Ciemny biały
        {"lower": np.array([0, 0, 240], dtype=np.uint8), "upper": np.array([180, 30, 255], dtype=np.uint8)},
        # Jasny biały
    ],
    "czarny": [
        {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([180, 255, 50], dtype=np.uint8)},
        # Standardowy czarny
        {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([180, 255, 25], dtype=np.uint8)},
        # Ciemny czarny
        {"lower": np.array([0, 0, 50], dtype=np.uint8), "upper": np.array([180, 255, 80], dtype=np.uint8)},
        # Jasny czarny
    ],
    "szary": [
        {"lower": np.array([0, 0, 50], dtype=np.uint8), "upper": np.array([180, 20, 200], dtype=np.uint8)},
        # Standardowy szary
        {"lower": np.array([0, 0, 30], dtype=np.uint8), "upper": np.array([180, 20, 150], dtype=np.uint8)},
        # Ciemny szary
        {"lower": np.array([0, 0, 200], dtype=np.uint8), "upper": np.array([180, 20, 255], dtype=np.uint8)},
        # Jasny szary
    ],
    "turkusowy": [
        {"lower": np.array([80, 150, 100], dtype=np.uint8), "upper": np.array([100, 255, 255], dtype=np.uint8)},
        # Standardowy turkusowy
        {"lower": np.array([80, 150, 50], dtype=np.uint8), "upper": np.array([100, 255, 150], dtype=np.uint8)},
        # Ciemny turkusowy
        {"lower": np.array([80, 150, 200], dtype=np.uint8), "upper": np.array([100, 255, 255], dtype=np.uint8)},
        # Jasny turkusowy
    ],
    "błękit": [
        {"lower": np.array([90, 150, 100], dtype=np.uint8), "upper": np.array([110, 255, 255], dtype=np.uint8)},
        # Standardowy błękit
        {"lower": np.array([90, 150, 50], dtype=np.uint8), "upper": np.array([110, 255, 150], dtype=np.uint8)},
        # Ciemny błękit
        {"lower": np.array([90, 150, 200], dtype=np.uint8), "upper": np.array([110, 255, 255], dtype=np.uint8)},
        # Jasny błękit
    ],
}


# ==================================================================================================================
# Sekcja 
# ==================================================================================================================

def draw_bboxes_on_image(image, matching_bboxes):
    for bbox_data in matching_bboxes:
        bbox = bbox_data["bbox"]
        x1, y1, width, height = map(int, bbox)
        x2 = x1 + width
        y2 = y1 + height

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        color_info = bbox_data["colors"]
        text = ', '.join([f"{color}: {info['percentage']:.1f}%" for color, info in color_info.items() if info["found"]])
        cv2.putText(image,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    return image


def draw_bbox_on_image_2(image: np.ndarray, matching_bbox: Detection) -> np.ndarray:
    x1 = matching_bbox.x
    y1 = matching_bbox.y
    width = matching_bbox.width
    height = matching_bbox.height
    x2 = x1 + width
    y2 = y1 + height

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    #color_info = bbox_data["colors"]
    #text = ', '.join([f"{color}: {info['percentage']:.1f}%" for color, info in color_info.items() if info["found"]])
    text = f'confidence: {matching_bbox.confidence:.2f}%'
    cv2.putText(image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

    return image


# ==================================================================================================================
# Sekcja 
# ==================================================================================================================
def all_images_threading(images_data, color_names, min_color_perc):
    result_queue = queue.Queue()
    threads = []

    for every_image in images_data:
        thread = threading.Thread(target=process_images_in_folder,
                                  args=(every_image, color_names, min_color_perc, result_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    matching_images = []
    while not result_queue.empty():
        matching_images.append(result_queue.get())

    # print(f"Ile przeszło: {len(matching_images)} z {len(images_data)}")
    return matching_images


# ==================================================================================================================
# Sekcja 
# ==================================================================================================================


def process_images_in_folder(every_image, color_names, min_color_perc, result_queue):
    model = YOLO(YOLO11_MODEL_PATH, verbose=False)
    start_image_time = time.time()

    image = every_image["image"]
    bboxes = every_image["bboxes"]
    image_name = every_image.get("name", "Unknown")

    matching_bboxes = []

    for bbox in bboxes:
        x1, y1, width, height = bbox
        x1, y1, width, height = map(int, [x1, y1, width, height])
        x2 = x1 + width
        y2 = y1 + height
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        cropped_image = image[y1:y2, x1:x2]

        results = model(cropped_image, classes=[0], verbose=False)

        for result in results:
            masks = result.masks

            if masks is not None:
                mask = masks[
                    0].data.cpu().numpy()  ## YOLOv8 zwraca maski jako tensory PyTorcha. Aby pracować z nimi w OpenCV, musimy je przekonwertować do formatu NumPy
                mask = (mask * 255).astype(np.uint8)  ## format 0,255
                mask = np.squeeze(mask)  ## YOLO maski daje w 3 wymirach a my w OpenCV 2 potrzebujemy bo obrazek

                if len(mask.shape) > 2:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                mask_resized = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))
                black_background = np.zeros_like(cropped_image)

                cleaned_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_resized)

                all_color_ranges = {
                    color_name: color_by_name(color_name)
                    for color_name in color_names if color_by_name(color_name)
                }

                color_info, total_color_percentage, result_image = calculate_color_percentage(cleaned_image,
                                                                                              all_color_ranges,
                                                                                              min_color_perc)

                if all(info["found"] for info in color_info.values()):
                    matching_bboxes.append({
                        "bbox": bbox,
                        "colors": color_info,
                        "total_color_percentage": total_color_percentage
                    })

    image_with_bboxes = draw_bboxes_on_image(image.copy(), matching_bboxes)

    result_data = {
        "image_name": image_name,
        "matching_bboxes": matching_bboxes,
        "processing_time": time.time() - start_image_time,
        "image_with_bboxes": image_with_bboxes
    }
    result_queue.put(result_data)


# ==================================================================================================================
# Sekcja 2: 
# ==================================================================================================================
def calculate_color_percentage(image, color_ranges, min_color_perc):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  ## skoro ranges są w HSV to musimy zamienic image tez w to
    total_pixels = image.shape[0] * image.shape[1]  ## liczymy wielkosc obrazka
    non_black_pixels = np.sum(np.all(hsv_image != [0, 0, 0], axis=-1))  # Piksele, które nie są czarne

    color_info = {}
    total_color_pixels = 0
    result_image = image.copy()

    # Tworzenie masek dla kolejnych kolorow
    for color_name, ranges in color_ranges.items():
        color_pixels = 0
        for color_range in ranges:
            lower = color_range["lower"]
            upper = color_range["upper"]
            mask = cv2.inRange(hsv_image, lower, upper)  ## Sprawdzamy czy w granicach ranges

            color_pixels += np.sum(mask > 0)  ## Dodjemy sobie pixcele spelniajace to

            colored_area = cv2.bitwise_and(result_image, result_image, mask=mask)  ## Obrazek z tylko z pixelami koloru
            result_image = cv2.add(result_image, colored_area)

        color_percentage = (color_pixels / non_black_pixels) * 100
        color_info[color_name] = {
            "percentage": color_percentage,
            "found": color_percentage >= min_color_perc.get(color_name, 0),
            "pixel_count": color_pixels
        }

        total_color_pixels += color_pixels

    total_color_percentage = (total_color_pixels / non_black_pixels) * 100

    return color_info, total_color_percentage, result_image


# ==================================================================================================================
# Sekcja 3: 
# ==================================================================================================================
def color_by_name(color_name):
    color_name = color_name.lower()
    if color_name in COLOR_RANGES:
        return COLOR_RANGES[color_name]
    else:
        print(f"Nieznany kolor: {color_name}. Dostępne kolory: {', '.join(COLOR_RANGES.keys())}")
        return None


# ==================================================================================================================
# Sekcja 4: 
# ==================================================================================================================

## INPUTY image POŹNIEJ OUT
#if __name__ == "__main__":
def color_detection(images_data):
    # Pobieranie danych wejściowych
    # images_data = [
    #     {
    #         "image": cv2.imread("1_test.jpeg"),
    #         'bboxes': [(756.31, 492.04, 77.04644775390625, 181.96072387695312),
    #                    (505.71, 553.78, 104.32180786132812, 120.22113037109375),
    #                    (382.73, 464.84, 88.91354370117188, 189.35989379882812),
    #                    (133.06, 353.38, 72.82888793945312, 175.06396484375),
    #                    (592.95, 549.75, 88.4273681640625, 123.02349853515625),
    #                    (269.82, 366.7, 46.93890380859375, 142.06512451171875),
    #                    (626.32, 295.18, 33.36297607421875, 98.61917114257812),
    #                    (439.59, 347.02, 46.29742431640625, 136.87625122070312)],
    #         "name": "1_test.jpeg"
    #     },
    #     {
    #         "image": cv2.imread("2_test.jpeg"),
    #         'bboxes': [(744.84, 495.27, 266.52197265625, 541.053466796875),
    #                    (198.51, 552.72, 424.4604187011719, 339.08843994140625),
    #                    (1012.01, 440.8, 89.3568115234375, 183.63824462890625),
    #                    (1216.19, 420.38, 75.7652587890625, 151.48333740234375),
    #                    (748.08, 290.69, 99.38348388671875, 250.12686157226562),
    #                    (277.9, 235.9, 54.19891357421875, 144.4530792236328),
    #                    (434.75, 246.15, 75.09201049804688, 198.70315551757812),
    #                    (1151.98, 602.73, 223.920166015625, 328.7281494140625),
    #                    (1115.0, 566.89, 168.178955078125, 340.10260009765625),
    #                    (387.32, 214.02, 65.8631591796875, 182.76942443847656),
    #                    (921.56, 357.29, 68.66290283203125, 172.58700561523438),
    #                    (1161.05, 285.15, 56.3072509765625, 95.80172729492188),
    #                    (527.92, 237.17, 53.4163818359375, 223.22528076171875),
    #                    (67.27, 181.09, 67.05166625976562, 172.31402587890625),
    #                    (817.65, 295.73, 71.25360107421875, 141.949951171875),
    #                    (223.65, 265.14, 43.2935791015625, 89.47079467773438),
    #                    (149.01, 209.6, 48.929656982421875, 141.76458740234375),
    #                    (445.69, 236.92, 55.306365966796875, 199.06541442871094)],
    #         "name": "2_test.jpeg"
    #     }
    # ]

    color_names_input = input("Podaj kolory (oddzielone przecinkami): ")
    color_names = [color.strip() for color in color_names_input.split(",")]

    min_color_perc = {}
    for color in color_names:
        while True:
            try:
                min_per_color = float(input(f"Minimum dla {color} (%): "))
                min_color_perc[color] = min_per_color
                break
            except:
                break

    # Wywołanie głównej funkcji przetwarzającej
    matching_images = all_images_threading(images_data, color_names, min_color_perc)
    # print("Obrazy spełniające kryteria:", matching_images)

    for result in matching_images:
        image_with_bboxes = result["image_with_bboxes"]

        # Wyświetl obrazek
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Obrazek: {result['image_name']}")
        plt.axis('off')
        plt.show()
