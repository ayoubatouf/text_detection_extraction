import cv2
from extract_text.output_saver import OutputSaver


class FileOutputSaver(OutputSaver):
    def save_image(self, image, image_path):
        cv2.imwrite(image_path, image)

    def save_text(self, text, text_path):
        with open(text_path, "w") as file:
            file.write(" ".join(text))
