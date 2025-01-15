import cv2
import numpy as np
import gc
from typing import List, Tuple
from filter_images.i_text_detection_model import ITextDetectionModel


class TextDetector(ITextDetectionModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.east_net = self.load_east_model()

    def load_east_model(self) -> cv2.dnn_Net:
        try:
            return cv2.dnn.readNet(self.model_path)
        except cv2.error as e:
            raise RuntimeError(f"Failed to load EAST model: {e}")

    def detect_text(
        self, image: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        original_h, original_w = image.shape[:2]
        new_h, new_w = max((original_h // 32) * 32, 320), max(
            (original_w // 32) * 32, 320
        )
        resized_image = (
            cv2.resize(image, (new_w, new_h))
            if (original_h != new_h or original_w != new_w)
            else image
        )

        blob = cv2.dnn.blobFromImage(
            resized_image,
            1.0,
            (new_w, new_h),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self.east_net.setInput(blob)
        scores, geometry = self.east_net.forward(
            ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        )

        rectangles, confidences = self.decode_predictions(
            scores, geometry, conf_threshold
        )
        indices = cv2.dnn.NMSBoxes(
            rectangles, confidences, conf_threshold, nms_threshold
        )

        del blob, resized_image, scores, geometry
        gc.collect()

        return indices, rectangles, confidences

    def decode_predictions(
        self, scores, geometry, conf_threshold: float
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        num_rows, num_cols = scores.shape[2:4]
        rectangles, confidences = [], []

        for y in range(num_rows):
            for x in range(num_cols):
                score = scores[0, 0, y, x]
                if score < conf_threshold:
                    continue
                offset_x, offset_y = x * 4, y * 4
                angle = geometry[0, 4, y, x]
                cos, sin = np.cos(angle), np.sin(angle)
                h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
                w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
                end_x = int(
                    offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x]
                )
                end_y = int(
                    offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x]
                )
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                rectangles.append((start_x, start_y, w, h))
                confidences.append(float(score))

        del geometry
        gc.collect()
        return rectangles, confidences
