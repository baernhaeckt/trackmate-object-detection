import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetectionService:
    def __init__(self):
        self.object_detection_model = YOLO("./../model/yolov10n.pt")

    @staticmethod
    def _distance_calculator(box, img: np.ndarray) -> float:
        _, width, _ = img.shape

        object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item() * 0.37
        distance = (width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)

        return round(distance, 2)

    def predict_and_detect(self, file_path: str) -> list:
        bounding_boxes = list()

        image = cv2.imread(file_path)
        results = self.object_detection_model.predict(image, conf=0.7, imgsz=(384, 480))
        for result in results:
            for box in result.boxes:
                # get the coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # calculate distance
                distance = self._distance_calculator(box, image)

                bounding_boxes.append({
                    "coordinates": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "distance": distance,
                })
        return bounding_boxes
