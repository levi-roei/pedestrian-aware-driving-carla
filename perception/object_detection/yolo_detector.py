from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import Optional, List, Dict
from .base_object_detector import BaseObjectDetector


class YOLODetector(BaseObjectDetector):
    def __init__(
        self,
        weights_path: str,
        classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None
    ):
        """
        YOLO-based object detector.

        Parameters:
        - weights_path: path of the YOLO weights to load.
        - classes: Optional list of class names to filter output.
        - confidence_threshold: Optional confidence threshold to filter output.
        """
        model = YOLO(weights_path)
        super().__init__(model=model, classes=classes, confidence_threshold=confidence_threshold)

    def _detect(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Perform object detection on the input image using YOLO.

        Parameters:
        - image: PIL.Image instance.

        Returns:
        - A dictionary with:
            - 'boxes': Nx4 NumPy array of bounding boxes [x1, y1, x2, y2]
            - 'class_ids': NumPy array of class IDs (floats, converted to ints in detect)
            - 'confidences': NumPy array of confidence scores (floats)
            - 'label_map': Dictionary mapping int class IDs to string labels
        """
        results = self._model.predict(image, verbose=False)[0]
        boxes = results.boxes

        return {
            'boxes': boxes.xyxy.cpu().numpy(),
            'class_ids': boxes.cls.cpu().numpy(),
            'confidences': boxes.conf.cpu().numpy(),
            'label_map': results.names
        }
