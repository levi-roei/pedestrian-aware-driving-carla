from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from sensors import SensorManager
from motion_estimation import Motion


class BaseObjectsMotionEstimator(ABC):
    """
    Abstract base class for multi-object motion estimators.

    Subclasses must implement `_estimate_objects_motion`, which computes object
    motions using named sensor data inputs. This class handles retrieving sensor
    data from the provided sensor manager using a configured mapping and timeout.
    """

    def __init__(
        self,
        sensor_manager: SensorManager,
        sensor_config: Dict[str, str],
        get_data_timeout: float = 10.0
    ) -> None:
        """
        Initialize the motion estimator with a sensor manager and sensor configuration.

        Parameters
        ----------
        sensor_manager : SensorManager
            Provides access to sensor data (must implement get_data()).
        sensor_config : Dict[str, str]
            Mapping from argument names (expected by `_estimate_objects_motion`)
            to sensor names used in the sensor manager.
        get_data_timeout : float, optional
            Timeout for sensor data retrieval in seconds. Defaults to 10.0.
        """
        self._sensor_manager = sensor_manager
        self._sensor_config = sensor_config
        self._get_data_timeout = get_data_timeout

    def estimate_objects_motion(
        self,
        simulation_time: Optional[float] = None
    ) -> Dict[str, Motion]:
        """
        Estimate the motion of all tracked objects.

        Retrieves sensor data using the configured mapping and forwards it
        (along with an optional `simulation_time`) to `_estimate_objects_motion`.

        Parameters
        ----------
        simulation_time : Optional[float], keyword-only
            If provided, passes the current simulation time in seconds to
            `_estimate_objects_motion` as an extra keyword argument.

        Returns
        -------
        Dict[str, Motion]
            Mapping from object ID to its motion.
        """
        sensor_names = list(self._sensor_config.values())
        sensor_data_by_name = self._sensor_manager.get_data(
            sensor_names, self._get_data_timeout
        )

        # Build the sensor args from the mapping
        sensor_args = {
            arg_name: sensor_data_by_name[sensor_name]
            for arg_name, sensor_name in self._sensor_config.items()
        }

        # Forward simulation_time if provided
        if simulation_time is not None:
            return self._estimate_objects_motion(**sensor_args, simulation_time=simulation_time)
        else:
            return self._estimate_objects_motion(**sensor_args)

    @abstractmethod
    def _estimate_objects_motion(
        self,
        **sensor_data: Any
    ) -> Dict[str, Motion]:
        """
        Abstract method to estimate object motions using provided sensor data.

        Parameters
        ----------
        **sensor_data : Any
            Keyword arguments as defined in `sensor_config`, plus optionally:
              • simulation_time: float (current simulation time in seconds).

        Returns
        -------
        Dict[str, Motion]
            Mapping from object ID to its motion.
        """
        raise NotImplementedError
