from paddleocr import PaddleOCR
from extract_text.ocr_processor import OCRProcessor


class PaddleOCRProcessor(OCRProcessor):
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def process(self, image_path):
        return self.ocr.ocr(image_path, cls=True)
