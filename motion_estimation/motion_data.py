from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class MotionEstimationInput:
    """
    Input data for motion estimation.

    Attributes:
        position (Optional[np.ndarray]): The observed position, typically a 3D vector.
        position (Optional[np.ndarray]): The observed velocity, typically a 3D vector.
    """
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None


@dataclass
class Motion:
    """
    Output data from a motion estimator.

    Attributes:
        position (Optional[np.ndarray]): The estimated position.
        velocity (Optional[np.ndarray]): The estimated velocity vector.
        speed (Optional[float]): The scalar speed, i.e., the magnitude of the velocity vector.
        path (Optional[List[np.ndarray]]): A list of estimated positions (e.g. past or predicted trajectory).
        path_times (Optional[List[float]]): Corresponding timestamps for each point in the path.
    """
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    speed: Optional[float] = None
    path: Optional[List[np.ndarray]] = None
    path_times: Optional[List[float]] = None
