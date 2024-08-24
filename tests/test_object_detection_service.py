import cv2
from services.object_detection_service import ObjectDetectionService


class TestObjectDetectionService:
    def test_object_detection_service(self):
        # arrange
        img_path = "./Bern_bahnhof002.jpg"
        img = cv2.imread(img_path)
        object_detection_service = ObjectDetectionService()

        # act
        bounding_boxes = object_detection_service.predict_and_detect(img_path)

        # assert
        for bounding_box in bounding_boxes:
            x1, y1, x2, y2 = bounding_box["coordinates"]
            class_name = bounding_box["class_name"]
            distance = bounding_box["distance"]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Class: {class_name}, Distance {distance}", (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imwrite("result.jpg", img)
