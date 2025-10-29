from abc import ABC, abstractmethod
from typing import Optional, Dict


from motion_estimation import Motion
from utils import EWMA


class BaseCollisionPredictor(ABC):
    """
    Abstract base class for predicting potential collisions between an ego vehicle
    and multiple moving objects.

    Subclasses must implement :meth:`earliest_time_to_object_collision` to provide
    the collision prediction logic for a single object.
    """

    def __init__(self,
                 safety_threshold: float,
                 ttc_alpha: float = 0.1) -> None:
        """
        Initialize the collision predictor.

        Parameters
        ----------
        safety_threshold : float
            Minimum allowed distance (in meters) between the ego vehicle and another object
            before it is considered a collision.
        ttc_alpha : float, optional
            Smoothing factor (0–1) for the EWMA filter used on TTC values.
            Default is 0.1.
        """
        self._safety_threshold = safety_threshold
        self._min_ttc = float("inf")
        self._ttc_given_coll_ewma = EWMA(ttc_alpha)

    def earliest_time_to_objects_collision(
        self,
        ego_motion: Motion,
        objects_motion: Dict[str, Motion]
    ) -> Dict[str, Optional[float]]:
        """
        Compute the earliest collision time between the ego vehicle and any object.

        Parameters
        ----------
        ego_motion : Motion
            Motion estimation data for the ego vehicle.
        objects_motion : Dict[str, Motion]
            A dictionary mapping object IDs (str) to their respective motion data.

        Returns
        -------
        Dict[str, Optional[float]]
            A dictionary containing:
            - `"time"`: The earliest time to collision (float) or None if no collision is expected.
            - `"id"`: The ID (str) of the object with the earliest collision or None if no collision.
        """
        min_time = None
        min_id = None

        for obj_id, obj_motion in objects_motion.items():
            t_contact = self.earliest_time_to_object_collision(ego_motion, obj_motion)
            if t_contact is not None and (min_time is None or t_contact < min_time):
                min_time = t_contact
                min_id = obj_id

        if min_time is not None:
            self._ttc_given_coll_ewma.update(min_time)
            if min_time < self._min_ttc:
                self._min_ttc = min_time
            
        return {
            "time": min_time,
            "id": min_id
        }

    @abstractmethod
    def earliest_time_to_object_collision(
        self,
        ego_motion: Motion,
        object_motion: Motion
    ) -> Optional[float]:
        """
        Compute the earliest time of collision between the ego vehicle and a single object.

        This method must be implemented by subclasses.

        Parameters
        ----------
        ego_motion : Motion
            Motion estimation data for the ego vehicle.
        object_motion : Motion
            Motion estimation data for the other object.

        Returns
        -------
        Optional[float]
            The earliest time to collision (in seconds) or None if no collision is expected.
        """
        raise NotImplementedError

    def get_estimated_ttc_stats(self) -> Dict[str, Optional[float]]:
        """
        Retrieve statistics on the time-to-collision (TTC) observed so far.

        Returns
        -------
        Dict[str, Optional[float]]
            A dictionary containing:
                * `"mean"` : Optional[float]
                    The EWMA-smoothed TTC over all observed collisions, or None if no data yet.
                * `"min"` : float
                    The smallest TTC observed so far, or `float('inf')` if none.
        """
        return {'mean': self._ttc_given_coll_ewma.get_avg(),
                'min': self._min_ttc}
