import cv2
import os
from extract_text.bounding_box_drawer import BoundingBoxDrawer
from extract_text.ocr_processor import OCRProcessor
from extract_text.output_saver import OutputSaver
from extract_text.text_line_processor import TextLineProcessor


class TextExtractor:
    def __init__(
        self,
        ocr_processor: OCRProcessor,
        output_saver: OutputSaver,
        line_processor: TextLineProcessor,
        drawer: BoundingBoxDrawer,
    ):
        self.ocr_processor = ocr_processor
        self.output_saver = output_saver
        self.line_processor = line_processor
        self.drawer = drawer

    def extract_from_image(self, image_path, text_folder, bounds_folder):
        output_image_path, output_text_path = self._get_output_paths(
            image_path, text_folder, bounds_folder
        )
        result = self.ocr_processor.process(image_path)

        if not result or not result[0]:
            print(f"No text detected in {image_path}. Skipping.")
            return ""

        image = cv2.imread(image_path)
        text_lines = self.line_processor.process(result)
        sorted_lines = self.line_processor.sort(text_lines)
        extracted_text = [text for _, text in sorted_lines]

        self.drawer.draw(image, sorted_lines)
        self.output_saver.save_image(image, output_image_path)
        self.output_saver.save_text(extracted_text, output_text_path)

        del image

        return " ".join(extracted_text)

    def _get_output_paths(self, image_path, text_folder, bounds_folder):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(bounds_folder, f"{base_name}_output.jpg")
        output_text_path = os.path.join(text_folder, f"{base_name}_extracted_text.txt")
        return output_image_path, output_text_path
