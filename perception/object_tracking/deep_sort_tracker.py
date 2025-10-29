from deep_sort.deep_sort.tracker import Tracker as DeepSortInternalTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from typing import Dict, List
from PIL import Image
import numpy as np

from .base_object_tracker import BaseObjectTracker
from ..utils import clip_and_int_boxes


class DeepSortTracker(BaseObjectTracker):
    """
    Object tracker using Deep SORT algorithm with optional configuration.

    Parameters:
    - encoder_weights_path (str): Path to the feature encoder model sfile.
    - metric (NearestNeighborDistanceMetric): Distance metric used for data association. If None, uses cosine with 0.4 threshold.
    """

    def __init__(
        self,
        encoder_weights_path: str,
        metric: nn_matching.NearestNeighborDistanceMetric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4)
    ):
        self._encoder = gdet.create_box_encoder(encoder_weights_path, batch_size=1)
        self._tracker = DeepSortInternalTracker(metric)
        self._tracked_objects = {}

    def update(self, image: Image.Image, boxes: np.ndarray, confidences: np.ndarray) -> None:
        """
        Update the tracker with a new frame and detection results.

        Parameters:
        - image: PIL.Image of the current frame.
        - boxes: Nx4 NumPy array of bounding boxes [x1, y1, x2, y2].
        - confidences: NumPy array of confidence scores.
        """
        frame = np.array(image)

        if len(boxes) == 0:
            self._tracker.predict()
            self._tracker.update([])
            self._update_tracked_objects()
            return

        # Convert to [x, y, w, h]
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]

        features = self._encoder(frame, boxes_xywh)

        detections = [
            Detection(bbox, confidences[i], features[i])
            for i, bbox in enumerate(boxes_xywh)
        ]

        self._tracker.predict()
        self._tracker.update(detections)
        self._update_tracked_objects()


        image_w, image_h = image.size
        self._tracked_objects = {
            object_id: clip_and_int_boxes(box.reshape(1, -1), image_w, image_h).reshape(-1)
            for object_id, box in self._tracked_objects.items()
        }

    def _update_tracked_objects(self) -> None:
        """
        Internal method to update the confirmed and active tracks.
        """
        self._tracked_objects = {
            track.track_id: track.to_tlbr()
            for track in self._tracker.tracks
            if track.is_confirmed() and track.time_since_update <= 1
        }

    def get_tracked_objects(self) -> Dict[int, List[float]]:
        """
        Get the currently tracked objects.

        Returns:
        - Dictionary mapping object IDs to bounding boxes [x1, y1, x2, y2].
        """
        return {track_id: bbox.tolist() for track_id, bbox in self._tracked_objects.items()}
