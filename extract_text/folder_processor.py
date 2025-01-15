import os
from extract_text.text_extractor import TextExtractor


class FolderProcessor:
    def __init__(
        self, image_folder, text_folder, bounds_folder, text_extractor: TextExtractor
    ):
        self.image_folder = image_folder
        self.text_folder = text_folder
        self.bounds_folder = bounds_folder
        self.text_extractor = text_extractor

    def process(self):
        if not os.path.exists(self.text_folder):
            os.makedirs(self.text_folder)
        if not os.path.exists(self.bounds_folder):
            os.makedirs(self.bounds_folder)

        for filename in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                extracted_text = self.text_extractor.extract_from_image(
                    image_path, self.text_folder, self.bounds_folder
                )
                if extracted_text:
                    print(
                        f"Processed {filename}, extracted text saved to {self.text_folder} and bounding boxes saved to {self.bounds_folder}"
                    )
                else:
                    print(f"Skipping {filename} as no text was detected.")
