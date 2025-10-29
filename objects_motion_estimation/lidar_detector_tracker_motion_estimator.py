from typing import Dict, Any, Optional, List, Type

import numpy as np

from .base_objects_motion_estimator import BaseObjectsMotionEstimator
from motion_estimation import (
    BaseMotionEstimator,
    MotionEstimationInput,
    Motion)
from sensors import SensorManager
from perception.object_detection import BaseObjectDetector
from perception.object_tracking import BaseObjectTracker


class LidarDetectorTrackerMotionEstimator(BaseObjectsMotionEstimator):
    """
    Estimate and track the 3D motion (position and velocity) of multiple objects in a CARLA simulation.

    This class combines:
      • An image-based object detector,
      • A tracker assigning stable IDs to detections,
      • LiDAR point cloud data aligned to the camera image,
      • A per-object motion estimator class (e.g., the Kalman filter class).

    For each object ID provided by the tracker, a separate motion estimator instance is maintained.
    Estimators are updated with the mean LiDAR position of the object at each frame.
    If an object is not observed for longer than `timeout` seconds, its estimator is discarded.

    Parameters
    ----------
    sensor_manager : SensorManager
        A sensor manager that provides access to sensors defined in `sensor_config`.
    sensor_config : Dict[str, str]
        Must contain exactly the keys `'image'` and `'lidar'`, mapping to sensor names known to the sensor manager.
    detector : BaseObjectDetector
        Object detector with a `.detect(image)` method returning a dict with `'boxes'` and `'confidences'`.
    tracker : BaseObjectTracker
        Tracker with `.update(image, boxes, confidences)` and `.get_tracked_objects()` methods.
    motion_estimator_cls : Type[BaseMotionEstimator]
        **The class itself**, not an instance. This subclass of `BaseMotionEstimator` will be instantiated per tracked object.
    get_data_timeout : float, optional
        Timeout for waiting on sensor data (seconds). Defaults to 10.0.
    motion_estimator_args : Optional[List[Any]], optional
        Positional arguments passed to each `motion_estimator_cls` instance. Defaults to empty list.
    motion_estimator_kwargs : Optional[Dict[str, Any]], optional
        Keyword arguments passed to each `motion_estimator_cls` instance. Defaults to empty dict.
    timeout : float, optional
        If an object is not seen for this many seconds, its motion estimator is removed. Defaults to 1.0.

    Notes
    -----
    Each estimator is expected to require only the `position` field in `MotionEstimationInput`.
    The output of each estimator must include `position` and `velocity`.
    """

    def __init__(
        self,
        sensor_manager: SensorManager,
        sensor_config: Dict[str, str],
        detector: BaseObjectDetector,
        tracker: BaseObjectTracker,
        motion_estimator_cls: Type[BaseMotionEstimator],
        get_data_timeout: float = 10.0,
        motion_estimator_args: Optional[List[Any]] = None,
        motion_estimator_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 1.0
    ) -> None:
        """
        Initialize the LidarDetectorTrackerMotionEstimator.

        This sets up internal references to the detector, tracker, world,
        and the motion estimator class used per object. It also initializes
        internal state for managing motion estimator instances and last-seen times.

        Parameters
        ----------
        sensor_manager : SensorManager
            Sensor manager responsible for retrieving image and LiDAR data.
        sensor_config : Dict[str, str]
            Must contain `'image'` and `'lidar'` keys mapping to sensor names.
        detector : BaseObjectDetector
            Detector used to obtain bounding boxes and confidences for each frame.
        tracker : BaseObjectTracker
            Tracker that assigns stable IDs to detections over time.
        motion_estimator_cls : Type[BaseMotionEstimator]
            A subclass of `BaseMotionEstimator`. A new instance is created for each tracked object.
        get_data_timeout : float, optional
            Maximum time (seconds) to wait for sensor data. Defaults to 10.0.
        motion_estimator_args : Optional[List[Any]], optional
            Positional arguments for constructing each motion estimator. Defaults to empty.
        motion_estimator_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for constructing each motion estimator. Defaults to empty.
        timeout : float, optional
            Time in seconds after which an unseen object is considered stale and removed. Defaults to 1.0.
        """
        super().__init__(sensor_manager, sensor_config, get_data_timeout)

        self._detector = detector
        self._tracker = tracker
        self._timeout = timeout

        self._motion_estimator_cls = motion_estimator_cls
        self._estimator_args = motion_estimator_args or []
        self._estimator_kwargs = motion_estimator_kwargs or {}

        # Per-object state
        self._motion_estimators = {}
        self._last_seen = {}


    def _estimate_objects_motion(
        self,
        image: Any,
        lidar: Dict[str, np.ndarray],
        simulation_time: float
    ) -> Dict[str, Motion]:
        """
        Estimate and update the 3D motion (position & velocity) of each tracked object.

        Workflow
        --------
        1. Detect objects in the camera image.
        2. Update the tracker with current detections.
        3. For each tracked object, select LiDAR points that project inside its bounding box.
        4. Compute the mean 3D position of those points.
        5. Update or create a motion estimator for that object using the mean position.
        6. Remove stale estimators for objects not seen within `timeout` seconds.

        Parameters
        ----------
        image : Any
            The current image frame (as required by the detector and tracker).
        lidar : Dict[str, np.ndarray]
            Dictionary with:
              • `'world'`: np.ndarray of shape (N, 3) — 3D LiDAR points in world coordinates.
              • `'image'`: np.ndarray of shape (N, 3) — corresponding 2D pixel projections
                (x, y, <optional-depth>) aligned with the image.
        simulation_time : float
            The current simulation time.

        Returns
        -------
        Dict[str, Motion]
            Mapping from object ID to its motion.

        Raises
        ------
        ValueError
            If the detector or tracker return unexpected data structures.
        """
        if simulation_time is None:
            raise ValueError(
                "simulation_time must be provided to _estimate_objects_motion."
            )
        
        # Run detection and tracking
        detections = self._detector.detect(image)
        boxes = detections['boxes']
        confidences = detections['confidences']

        self._tracker.update(image, boxes, confidences)
        tracked_objects = self._tracker.get_tracked_objects()

        world_points = lidar['world']
        image_points = lidar['image']
        image_points_xy = image_points[:, :2]

        object_motions = {}

        # Process each tracked object
        for object_id, (x1, y1, x2, y2) in tracked_objects.items():
            mask = (
                (image_points_xy[:, 0] >= x1) & (image_points_xy[:, 0] <= x2) &
                (image_points_xy[:, 1] >= y1) & (image_points_xy[:, 1] <= y2)
            )
            object_3d_points = world_points[mask]
            if object_3d_points.shape[0] == 0:
                continue

            object_position = np.mean(object_3d_points, axis=0)

            # Create or update the object's motion estimator
            if object_id not in self._motion_estimators:
                estimator = self._motion_estimator_cls(
                    MotionEstimationInput(position=object_position, velocity=np.zeros(3)),
                    *self._estimator_args,
                    **self._estimator_kwargs
                )
                self._motion_estimators[object_id] = estimator
            else:
                estimator = self._motion_estimators[object_id]

            motion = estimator.estimate_motion(
                MotionEstimationInput(position=object_position)
            )
            object_motions[object_id] = motion
            self._last_seen[object_id] = simulation_time

        # Remove stale object estimators
        stale_ids = [
            obj_id for obj_id, t in self._last_seen.items()
            if simulation_time - t > self._timeout
        ]
        for obj_id in stale_ids:
            del self._last_seen[obj_id]
            del self._motion_estimators[obj_id]

        return object_motions
