from abc import ABC, abstractmethod


class OCRProcessor(ABC):
    @abstractmethod
    def process(self, image_path):
        pass
