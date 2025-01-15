from filter_images.i_image_processor import IImageProcessor
from filter_images.image_mover import ImageMover
from filter_images.image_processor import ImageProcessor
from filter_images.text_detector import TextDetector


class ImageProcessorFactory:
    def create_image_processor(
        self,
        source_folder: str,
        destination_folder: str,
        east_model_path: str,
        batch_size: int = 32,
    ) -> IImageProcessor:
        text_detector = TextDetector(east_model_path)
        image_mover = ImageMover(destination_folder)
        return ImageProcessor(
            source_folder, destination_folder, text_detector, image_mover, batch_size
        )
