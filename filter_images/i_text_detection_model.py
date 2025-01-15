import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class ITextDetectionModel(ABC):
    @abstractmethod
    def detect_text(
        self, image: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        pass
