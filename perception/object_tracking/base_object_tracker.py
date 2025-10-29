from abc import ABC, abstractmethod
from typing import Dict, List
from PIL import Image
import numpy as np


class BaseObjectTracker(ABC):
    @abstractmethod
    def update(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        confidences: np.ndarray
    ) -> None:
        """
        Update the tracker with a new frame and detected objects.

        Parameters:
        - image: PIL.Image of the current frame.
        - boxes: Nx4 NumPy array of bounding boxes [x1, y1, x2, y2].
        - confidences: NumPy array of confidence scores for each bounding box.

        Returns:
        - None
        """
        pass

    @abstractmethod
    def get_tracked_objects(self) -> Dict[int, List[float]]:
        """
        Get the currently tracked objects.

        Returns:
        - A dictionary mapping object IDs (int) to bounding boxes (list of 4 floats: [x1, y1, x2, y2]).
        """
        pass
