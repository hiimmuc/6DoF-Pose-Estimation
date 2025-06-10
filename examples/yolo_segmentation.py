import time

import cv2
from ultralytics import YOLO, solutions

TRACKING = True  # Set to True to enable tracking, False for segmentation only


def fps(start, end):
    return int(1 // (end - start))


model = YOLO("src/checkpoints/YOLO/yolo11n-seg.pt")  # load an official model

cap = cv2.VideoCapture(0)
try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Video frame is empty or processing is complete.")
            break

        start = time.perf_counter()
        if not TRACKING:
            results = model(
                image,
                conf=0.5,
                iou=0.5,
                # classes=[0]
            )  # Run inference
            # for result in results:
            # xy = result.masks.xy  # mask in polygon format
            # xyn = result.masks.xyn  # normalized
            # masks = result.masks.data  # mask in matrix format (num_objects x H x W)
            end = time.perf_counter()
            segments = results[0].plot()
            cv2.putText(
                segments,
                f"FPS: {fps(start, end)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Image Segmentation", segments)

        else:
            pass

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            print("Exit sequence initiated")
            break
except Exception as e:
    print(f"An error occurred: {repr(e)}")

finally:
    cap.release()
    cv2.destroyAllWindows()
