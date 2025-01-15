from extract_text.bounding_box_drawer import BoundingBoxDrawer
from extract_text.file_output_saver import FileOutputSaver
from extract_text.folder_processor import FolderProcessor
from extract_text.paddle_ocr_processor import PaddleOCRProcessor
from extract_text.text_extractor import TextExtractor
from extract_text.text_line_processor import TextLineProcessor


if __name__ == "__main__":
    image_folder = "images" # change this
    text_folder = "texts"
    bounds_folder = "bounds"

    ocr_processor = PaddleOCRProcessor(lang="en")
    output_saver = FileOutputSaver()
    line_processor = TextLineProcessor()
    drawer = BoundingBoxDrawer()

    text_extractor = TextExtractor(ocr_processor, output_saver, line_processor, drawer)
    folder_processor = FolderProcessor(
        image_folder, text_folder, bounds_folder, text_extractor
    )

    folder_processor.process()
