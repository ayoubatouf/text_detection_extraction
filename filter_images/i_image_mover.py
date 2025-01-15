from abc import ABC, abstractmethod


class IImageMover(ABC):
    @abstractmethod
    def move_image(self, image_path: str) -> float:
        pass
