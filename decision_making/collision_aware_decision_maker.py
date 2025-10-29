from typing import Dict, Optional

from motion_estimation import Motion, requires_motion_fields
from .collision_prediction import BaseCollisionPredictor
from .base_decision_maker import BaseDecisionMaker
from .utils import compute_reachable_path


class CollisionAwareDecisionMaker(BaseDecisionMaker):
    def __init__(
        self,
        target_speed: float,
        collision_predictor: BaseCollisionPredictor,
        collision_predictor_slowdown: BaseCollisionPredictor,
        collision_horizon_dist: float,
        emergency_stop_threshold: float = 0.5
    ):
        """
        Parameters
        ----------
        target_speed : float
            The normal desired speed for the agent.
        collision_predictor : BaseCollisionPredictor
            A collision predictor used to predict imminent collisions (for emergency stops).
        collision_predictor_slowdown : BaseCollisionPredictor
            A collision predictor used to predict collisions that require slowing down.
        collision_horizon_dist : float
            The distance horizon used for reachable path computation.
        emergency_stop_threshold : float
            Time threshold (seconds) below which an emergency stop is triggered.
        """
        self._target_speed = target_speed
        self._collision_predictor = collision_predictor
        self._collision_predictor_slowdown = collision_predictor_slowdown
        self._collision_horizon = collision_horizon_dist / self._target_speed
        self._emergency_stop_threshold = emergency_stop_threshold

    @requires_motion_fields(ego_motion=['path', 'speed'])
    def make_decision(self,
                      ego_motion: Motion,
                      objects_motion: Dict[str, Motion]) -> Dict:
        """
        Decide target speed and emergency stop based on predicted collisions.

        Parameters
        ----------
        ego_motion : Motion
            Contains ego vehicle path and current speed.
            **Important:** the `Motion` object passed to the collision predictors
            (after internal processing) will only include the fields `path` and `path_times`.
            Therefore, any fields required by a given `collision_predictor` or
            `collision_predictor_slowdown` must be a subset of these fields.
            In other words, the collision predictors must operate solely on `path` and `path_times`.

        objects_motion : Dict[str, Motion]
            Motion states of surrounding objects.
            **Important:** the `objects_motion` passed here is forwarded directly to the collision
            predictors. Any fields that the collision predictors require from `objects_motion`
            must be present in the `objects_motion` structure provided.

        Returns
        -------
        dict
            {
                "target_speed": float,
                "emergency_stop": bool
            }
        """
        reachable_path = compute_reachable_path(
            ego_motion.path,
            self._collision_horizon,
            ego_motion.speed
        )
        path = reachable_path["path"]
        path_times = reachable_path["path_times"]
        ego_motion_coll = Motion(path=path, path_times=path_times)

        # Predict earliest collision for emergency stop
        collision_prediction = self._collision_predictor.earliest_time_to_objects_collision(
            ego_motion_coll, objects_motion
        )
        time_to_collision = collision_prediction["time"]

        # Predict earliest collision for slowdown
        slowdown_prediction = self._collision_predictor_slowdown.earliest_time_to_objects_collision(
            ego_motion_coll, objects_motion
        )
        time_to_slowdown = slowdown_prediction["time"]

        # Decide on target speed
        if time_to_slowdown is None:
            target_speed = self._target_speed
        else:
            # Adjust target speed (0 if slowdown predictor sees a potential collision)
            # IMPORTANT NOTE: setting the target speed to 0, DOESN'T mean
            # the vehicle's speed will be zero in the next few time steps,
            # in practice it's essentially equivalent to hitting the "brake".
            target_speed = 0.0

        # Decide on emergency stop
        emergency_stop = (
            time_to_collision is not None and time_to_collision <= self._emergency_stop_threshold
        )

        return {
            "target_speed": target_speed,
            "emergency_stop": emergency_stop
        }
    
    def get_estimated_ttc_stats(self) -> Dict[str, Optional[float]]:
        """
        Retrieve statistics on the estimated time‑to‑collision (TTC)
        observed by this decision maker so far.

        Returns
        -------
        Dict[str, Optional[float]]
            A dictionary containing TTC metrics, for example:
                "mean" : Optional[float]
                    The smoothed or average TTC value in seconds,
                    or None if no collisions have been predicted yet.
                "min" : Optional[float]
                    The smallest TTC value observed (in seconds),
                    or None/float('inf') if no collision has been predicted.
        """
        return self._collision_predictor.get_estimated_ttc_stats()
