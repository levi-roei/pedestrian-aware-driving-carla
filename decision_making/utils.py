import numpy as np
from typing import List, Dict, Any


def compute_reachable_path(
    locations: List[np.ndarray],
    max_time: float,
    speed: float
) -> Dict[str, List[Any]]:
    """
    Compute the reachable portion of a path given time and speed constraints.

    Parameters
    ----------
    locations : List[np.ndarray]
        A sequence of 3D points (each a NumPy array of shape (3,))
        representing consecutive positions along a path.
    max_time : float
        The maximum travel time allowed, in seconds.
    speed : float
        Constant speed in meters per second.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary with:
        - "path": List of 3D NumPy arrays that are reachable within `max_time`.
        - "path_times": List of floats, each the cumulative time to reach
          the corresponding point in `path`.

    Notes
    -----
    If `locations` is empty or contains only one point, the function returns
    the same locations with time 0.0 for each.
    """
    if not locations or len(locations) < 2:
        return {
            "path": locations.copy(),
            "path_times": [0.0 for _ in locations]
        }

    traveled_time = 0.0
    path = [locations[0]]
    path_times = [0.0]

    for i in range(len(locations) - 1):
        p_current = locations[i]
        p_next = locations[i + 1]

        segment_vector = p_next - p_current
        segment_distance = float(np.linalg.norm(segment_vector))
        # Avoid division by zero
        segment_time = segment_distance / (speed + 1e-5)

        # If adding this segment would exceed the allowed time
        if traveled_time + segment_time >= max_time:
            remaining_time = max_time - traveled_time
            ratio = remaining_time / segment_time
            interpolated_point = p_current + ratio * segment_vector
            path.append(interpolated_point)
            path_times.append(max_time)
            break

        traveled_time += segment_time
        path.append(p_next)
        path_times.append(traveled_time)

    return {
        "path": path,
        "path_times": path_times
    }
