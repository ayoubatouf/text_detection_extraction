from abc import ABC, abstractmethod


class IImageProcessor(ABC):
    @abstractmethod
    def process_images(self) -> None:
        pass
