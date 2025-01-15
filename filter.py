from filter_images.image_processor_factory import ImageProcessorFactory


if __name__ == "__main__":
	# change paths
    source_folder = "/home/ayoub/Desktop/text_detector/source"
    destination_folder = "/home/ayoub/Desktop/text_detector/destination"
    east_model_path = "frozen_east_text_detection.pb"
    batch_size = 32

    image_processor_factory = ImageProcessorFactory()
    image_processor = image_processor_factory.create_image_processor(
        source_folder, destination_folder, east_model_path, batch_size
    )
    image_processor.process_images()
