from filter_images.i_image_mover import IImageMover
import os
import time
import shutil


class ImageMover(IImageMover):
    def __init__(self, destination_folder: str):
        self.destination_folder = destination_folder

    def move_image(self, image_path: str) -> float:
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)

        image_name = os.path.basename(image_path)
        destination_path = os.path.join(self.destination_folder, image_name)

        try:
            start_time = time.time()
            shutil.move(image_path, destination_path)
            move_time = time.time() - start_time
            return move_time
        except shutil.Error as e:
            raise RuntimeError(f"Error moving image {image_path}: {e}")
