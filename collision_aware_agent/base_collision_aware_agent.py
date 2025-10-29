from abc import ABC, abstractmethod
from typing import Optional, Dict

import carla


class BaseCollisionAwareAgent(ABC):
    """
    Abstract base class for a collision‑aware autonomous agent.

    Subclasses must implement all control and query methods defined here.
    """

    @abstractmethod
    def run_step(self) -> carla.VehicleControl:
        """
        Execute one control step and return the vehicle control command.

        Returns
        -------
        carla.VehicleControl
            The control command to apply to the ego vehicle.
        """
        pass

    @abstractmethod
    def set_destination(self, destination: carla.Location) -> None:
        """
        Set a new navigation destination.

        Parameters
        ----------
        destination : carla.Location
            The target location the agent should navigate to.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """
        Check whether the agent has reached its destination.

        Returns
        -------
        bool
            True if the agent has arrived at its destination, False otherwise.
        """
        pass

    @abstractmethod
    def get_estimated_ttc_stats(self) -> Dict[str, Optional[float]]:
        """
        Retrieve statistics on the estimated time‑to‑collision (TTC)
        observed so far.

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
        pass

    @abstractmethod
    def get_run_step_latency_avg(self) -> Optional[float]:
        """
        Get the smoothed average latency of the `run_step` loop.

        Returns
        -------
        Optional[float]
            The EWMA of recent run_step latencies, or None if no steps have run yet.
        """
        pass
