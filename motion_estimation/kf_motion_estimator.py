from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from .motion_data import MotionEstimationInput, Motion
from .motion_decorators import requires_motion_fields, produces_motion_fields
from .base_motion_estimator import BaseMotionEstimator


class KFMotionEstimator(BaseMotionEstimator):
    """
    Motion estimator using a 3D Kalman Filter.

    This class estimates position and velocity in 3D space using a linear
    Kalman filter with a 6‑state vector:
        [px, py, pz, vx, vy, vz].

    Interaction with decorators
    ---------------------------
    - The constructor requires initial position and velocity fields:
      enforced via ``@requires_motion_fields(initial_input=['position', 'velocity'])``.
    - The `estimate_motion` method:
        * Requires the `position` field in the input.
        * Produces non‑None `position` and `velocity` fields in the output.
    """

    @requires_motion_fields(initial_input=["position", "velocity"])
    def __init__(
        self,
        initial_input: MotionEstimationInput,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        P: float = 1000.0,
        R: float = 5.0,
        Q: float = 0.01,
        delta_t: float = 0.05,
    ) -> None:
        """
        Initialize the Kalman filter with initial position and velocity.

        Parameters
        ----------
        initial_input : MotionEstimationInput
            Input object with non‑None `position` (shape (3,)) and `velocity` (shape (3,)).
            These initialize the internal state vector.
        F : Optional[np.ndarray], optional
            State transition matrix (6×6). If None, a constant‑velocity model is used.
        H : Optional[np.ndarray], optional
            Measurement matrix (3×6). If None, position is directly observed.
        P : float, optional
            Initial covariance scaling factor. Defaults to 1000.0.
        R : float, optional
            Measurement noise covariance scaling factor. Defaults to 5.0.
        Q : float, optional
            Process noise covariance scaling factor. Defaults to 0.01.
        delta_t : float, optional
            Time step (s) for constant‑velocity model. Defaults to 0.05.

        Notes
        -----
        The state vector is ordered as [px, py, pz, vx, vy, vz].
        """
        self._model = KalmanFilter(dim_x=6, dim_z=3)
        dt = delta_t

        # Default constant-velocity model
        if F is None:
            F = np.eye(6)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt

        # Default measurement matrix: directly observe position
        if H is None:
            H = np.zeros((3, 6))
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        self._model.F = F
        self._model.H = H
        self._model.P *= P
        self._model.R *= R
        self._model.Q *= Q

        # Initialize state vector
        self._model.x = np.concatenate([initial_input.position, initial_input.velocity])

    @requires_motion_fields(input_data=["position"])
    @produces_motion_fields("position", "velocity")
    def estimate_motion(self, input_data: MotionEstimationInput) -> Motion:
        """
        Update the Kalman filter with a new position measurement and return estimated motion.

        Parameters
        ----------
        input_data : MotionEstimationInput
            Input object with a non‑None `position` (shape (3,)).

        Returns
        -------
        Motion
            Output with:
            - `position`: np.ndarray of shape (3,) (estimated position),
            - `velocity`: np.ndarray of shape (3,) (estimated velocity).

        Notes
        -----
        The Kalman filter performs a predict step followed by an update step.
        The returned arrays are copies of the filter's internal state.
        """
        self._model.predict()
        self._model.update(input_data.position)

        return Motion(
            position=self._model.x[:3].copy(),
            velocity=self._model.x[3:].copy()
        )
