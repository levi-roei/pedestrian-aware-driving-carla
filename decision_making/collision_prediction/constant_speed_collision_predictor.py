import numpy as np
from typing import Optional

from .base_collision_predictor import BaseCollisionPredictor
from motion_estimation import Motion, requires_motion_fields
from .utils import first_time_within_threshold


class ConstantSpeedCollisionPredictor(BaseCollisionPredictor):
    """
    Predicts collisions assuming both the ego vehicle and the object move at constant velocities.
    """

    @requires_motion_fields(ego_motion=['path', 'path_times'],
                            object_motion=['position', 'velocity'])
    def earliest_time_to_object_collision(
        self,
        ego_motion: Motion,
        object_motion: Motion
    ) -> Optional[float]:
        """
        Computes the earliest time of collision between ego and a single object using linear motion.

        Args:
            ego_motion (Motion): Ego vehicle's motion data.
            object_motion (Motion): Other object's motion data.

        Returns:
            float or None: Earliest collision time, or None if no collision is predicted.
        """
        path = ego_motion.path
        times = ego_motion.path_times
        obj_position = object_motion.position
        obj_velocity = object_motion.velocity

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            t1 = times[i]
            t2 = times[i + 1]
            
            # Skip degenerate segment
            if np.isclose(t2 - t1, 0): continue
            
            v_ego = (p2 - p1) / (t2 - t1)
            t_contact = first_time_within_threshold(
                p1, v_ego, obj_position, obj_velocity, t1, t2, self._safety_threshold
            )

            if t_contact is not None:
                return t_contact

        return None
