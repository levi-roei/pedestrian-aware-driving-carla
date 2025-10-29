from abc import ABC, abstractmethod
from typing import Dict, Optional

from motion_estimation import Motion


class BaseDecisionMaker(ABC):
    """
    Abstract base class for decision-making logic in a collision-aware agent.
    Any subclass must implement the make_decision method and optionally track
    collision prediction metrics such as minimum time-to-collision (TTC).
    """

    @abstractmethod
    def make_decision(
        self,
        ego_motion: Motion,
        objects_motion: Dict[str, Motion]
    ) -> Dict:
        """
        Decide on the target speed and emergency stop status.

        Parameters
        ----------
        ego_motion : Motion
            Information about the ego vehicle's current motion state.
        objects_motion : Dict[str, Motion]
            Motion states of surrounding objects.

        Returns
        -------
        dict
            {
                "target_speed": float,
                "emergency_stop": bool
            }
        """
        pass

    @abstractmethod
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
        pass
