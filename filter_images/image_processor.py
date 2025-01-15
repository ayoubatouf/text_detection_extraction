import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import gc
from typing import List
from filter_images.i_image_mover import IImageMover
from filter_images.i_image_processor import IImageProcessor
from filter_images.i_text_detection_model import ITextDetectionModel


class ImageProcessor(IImageProcessor):
    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        text_detector: ITextDetectionModel,
        image_mover: IImageMover,
        batch_size: int = 32,
    ):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.text_detector = text_detector
        self.image_mover = image_mover
        self.batch_size = batch_size

        self.num_images_processed = 0
        self.num_images_with_text = 0
        self.num_images_with_no_text = 0
        self.num_images_moved = 0
        self.confidences = []
        self.bounding_box_sizes = []
        self.move_times = []
        self.start_time = time.time()
        self.initial_folder_size = self.get_folder_size(self.source_folder)

    def process_images(self) -> None:
        image_files = [
            entry.path
            for entry in os.scandir(self.source_folder)
            if entry.is_file()
            and entry.name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print("No images found in the source folder.")
            return

        with tqdm(
            total=len(image_files),
            desc="Processing and Moving Images",
            unit="image",
            dynamic_ncols=True,
        ) as pbar:
            batch_images = []
            for image_path in image_files:
                batch_images.append(image_path)

                if (
                    len(batch_images) == self.batch_size
                    or image_path == image_files[-1]
                ):
                    self.process_batch(batch_images)
                    pbar.update(len(batch_images))
                    batch_images.clear()

        self.report_statistics()

    def process_batch(self, batch_images: List[str]) -> None:
        images = [cv2.imread(img_path) for img_path in batch_images]

        for image, image_path in zip(images, batch_images):
            if image is None:
                continue

            indices, rectangles, confidences = self.text_detector.detect_text(image)
            self.num_images_processed += 1

            if len(indices) > 0:
                self.num_images_with_text += 1
                try:
                    move_time = self.image_mover.move_image(image_path)
                except RuntimeError as e:
                    print(f"Skipping image {image_path}: {e}")
                    continue
                self.num_images_moved += 1
                self.move_times.append(move_time)
                self.confidences.extend(confidences)

                for (x, y, w, h) in rectangles:
                    self.bounding_box_sizes.append((w, h))
            else:
                self.num_images_with_no_text += 1

        del images
        gc.collect()

    def get_folder_size(self, folder_path: str) -> int:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for file_name in filenames:
                file_path = os.path.join(dirpath, file_name)
                total_size += os.path.getsize(file_path)
        return total_size

    def report_statistics(self) -> None:
        if self.num_images_processed == 0:
            print("No images were processed.")
            return

        total_time = time.time() - self.start_time
        avg_process_time = (
            total_time / self.num_images_processed
            if self.num_images_processed > 0
            else 0
        )
        avg_confidence = np.mean(self.confidences) if self.confidences else 0
        max_confidence = max(self.confidences) if self.confidences else 0
        avg_width, avg_height = (
            np.mean(self.bounding_box_sizes, axis=0)
            if self.bounding_box_sizes
            else (0, 0)
        )
        avg_move_time = np.mean(self.move_times) if self.move_times else 0
        final_folder_size = self.get_folder_size(self.source_folder)
        folder_size_change = final_folder_size - self.initial_folder_size

        print("\n--- Statistics ---")
        print(f"Number of images processed: {self.num_images_processed}")
        print(f"Number of images moved: {self.num_images_moved}")
        print(f"Average confidence score: {avg_confidence:.2f}")
        print(f"Maximum confidence score: {max_confidence:.2f}")
        print(
            f"Average bounding box size: Width={avg_width:.2f}, Height={avg_height:.2f}"
        )
        print(
            f"Images containing text rate: {(self.num_images_with_text / self.num_images_processed) * 100:.2f}%"
        )
        print(f"Average processing time per image: {avg_process_time:.2f} seconds")
        print(f"Number of images with no detected text: {self.num_images_with_no_text}")
        print(f"Folder size change: {folder_size_change} bytes")
        print(f"Average time for image movement: {avg_move_time:.2f} seconds")
