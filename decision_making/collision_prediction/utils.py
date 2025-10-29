import numpy as np


def first_time_within_threshold(p_A: np.ndarray,
                                v_A: np.ndarray,
                                p_B: np.ndarray,
                                v_B: np.ndarray,
                                t1: float,
                                t2: float,
                                threshold: float):
    """
    Computes the earliest time in the interval [t1, t2] at which two objects A and B 
    are within a given distance threshold from each other, assuming both objects 
    move at constant velocities.

    Each object follows linear motion:
        p_A(t) = p_A + v_A * t
        p_B(t) = p_B + v_B * t

    The function solves the inequality:
        ||p_A(t) - p_B(t)|| <= threshold

    Parameters:
        p_A (np.ndarray): Initial position vector of object A.
        v_A (np.ndarray): Constant velocity vector of object A.
        p_B (np.ndarray): Initial position vector of object B.
        v_B (np.ndarray): Constant velocity vector of object B.
        t1 (float): Start of the time interval to consider.
        t2 (float): End of the time interval to consider.
        threshold (float): Distance threshold to check against.

    Returns:
        float or None:
            The earliest time t ∈ [t1, t2] such that the distance between A and B 
            is less than or equal to `threshold`. Returns None if no such time exists 
            within the interval.
    """
    delta_p = p_A - p_B
    delta_v = v_A - v_B

    a = np.dot(delta_v, delta_v)
    b = 2 * np.dot(delta_p, delta_v)
    c = np.dot(delta_p, delta_p) - threshold**2

    # Handle constant distance (no relative motion)
    if np.isclose(a, 0):
        if c <= 0:
            return t1  # Always within distance
        else:
            return None  # Never within distance

    # Solve quadratic inequality: at^2 + bt + c <= 0
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None  # No real roots, never within threshold

    sqrt_disc = np.sqrt(discriminant)
    t_low = (-b - sqrt_disc) / (2 * a)
    t_high = (-b + sqrt_disc) / (2 * a)

    # We want the first time in [t1, t2] ∩ [t_low, t_high]
    t_start = max(t1, min(t_low, t_high))
    t_end = min(t2, max(t_low, t_high))

    if t_start <= t_end:
        return t_start  # Earliest time within threshold
    else:
        return None  # No overlap with [t1, t2]
