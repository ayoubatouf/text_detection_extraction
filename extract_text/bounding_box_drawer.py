import cv2


class BoundingBoxDrawer:
    @staticmethod
    def draw(image, text_lines):
        for box, _ in text_lines:
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
