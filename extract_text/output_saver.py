from abc import ABC, abstractmethod


class OutputSaver(ABC):
    @abstractmethod
    def save_image(self, image, image_path):
        pass

    @abstractmethod
    def save_text(self, text, text_path):
        pass
