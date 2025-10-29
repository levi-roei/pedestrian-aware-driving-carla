from abc import ABC, abstractmethod

from .motion_data import MotionEstimationInput, Motion


class BaseMotionEstimator(ABC):
    """
    Abstract base class for motion estimation modules.

    Subclasses must implement :meth:`estimate_motion`, which takes a
    MotionEstimationInput and returns a Motion.

    Interaction with decorators
    ---------------------------
    To enforce data‑integrity contracts, subclasses are expected to use the
    provided decorators on their concrete implementations:

    • Use ``@requires_motion_fields(...)`` on the subclass implementation
      to explicitly declare which fields on the input data must be non‑None.
      For example:
      >>> @requires_motion_fields(input_data=['position', 'velocity'])
      ... def estimate_motion(self, input_data): ...

    • Use ``@produces_motion_fields(...)`` on the subclass implementation
      to explicitly declare which fields on the returned
      Motion are guaranteed to be populated.
      For example:
      >>> @produces_motion_fields('path', 'path_times')
      ... def estimate_motion(self, input_data): ...

    These decorators will raise a ``ValueError`` at runtime if the contract
    is not fulfilled. This helps ensure that all upstream and downstream
    components can rely on required fields being present.

    Notes
    -----
    The base class itself does not enforce any particular fields; it is
    the responsibility of each concrete subclass to define its own input
    requirements and output guarantees using the decorators.
    """

    @abstractmethod
    def estimate_motion(self, input_data: MotionEstimationInput) -> Motion:
        """
        Estimate the motion based on input data.

        Parameters
        ----------
        input_data : MotionEstimationInput
            Input data structure containing fields such as current position,
            velocity, or other relevant state needed for motion estimation.

        Returns
        -------
        Motion
            Output object containing estimated motion fields such as path,
            path_times, or velocities.

        Raises
        ------
        NotImplementedError
            If called directly on the base class.
        """
        raise NotImplementedError
