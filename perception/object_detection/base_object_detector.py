from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
from ..utils import clip_and_int_boxes


class BaseObjectDetector(ABC):
    def __init__(
        self,
        model,
        classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None
    ):
        """
        Base class for object detection models.

        Parameters:
        - model: The underlying detection model.
        - classes: List of string labels to include in the output. If None, all detected classes are included.
        - confidence_threshold: Float threshold (0.0 to 1.0). Detections with confidence below this are excluded.
        """
        self._model = model
        self._classes = set(classes) if classes is not None else None
        self._confidence_threshold = confidence_threshold

    def detect(
        self,
        image: Image.Image
    ) -> Dict[str, np.ndarray]:
        """
        Detect objects in a given PIL image.

        Parameters:
        - image: PIL.Image instance.

        Returns:
        - A dictionary with:
            - 'boxes': Nx4 NumPy array of bounding boxes [x1, y1, x2, y2]
            - 'class_ids': NumPy array of class IDs
            - 'confidences': NumPy array of confidence scores
            - 'label_map': Dictionary mapping class ID (int) to class name (str)
        """
        output = self._detect(image)
        boxes = output['boxes']
        class_ids = output['class_ids']
        confidences = output['confidences']
        label_map = output['label_map']

        image_w, image_h = image.size
        boxes = clip_and_int_boxes(boxes, image_w, image_h)
        class_ids = class_ids.astype(int)

        # Apply class filtering
        if self._classes is not None:
            keep = [i for i, cid in enumerate(class_ids) if label_map[cid] in self._classes]
            boxes = boxes[keep]
            class_ids = class_ids[keep]
            confidences = confidences[keep]

        # Apply confidence threshold filtering
        if self._confidence_threshold is not None:
            keep = confidences >= self._confidence_threshold
            boxes = boxes[keep]
            class_ids = class_ids[keep]
            confidences = confidences[keep]

        return {
            'boxes': boxes,
            'class_ids': class_ids,
            'confidences': confidences,
            'label_map': label_map
        }

    @abstractmethod
    def _detect(
        self,
        image: Image.Image
    ) -> Dict[str, np.ndarray]:
        """
        Abstract method to be implemented by subclasses.

        Returns:
        - A dictionary with:
            - 'boxes': Nx4 NumPy array of bounding boxes [x1, y1, x2, y2]
            - 'class_ids': NumPy array of class IDs (ints)
            - 'confidences': NumPy array of confidence scores (floats)
            - 'label_map': Dictionary mapping int class IDs to string labels
        """
        pass
