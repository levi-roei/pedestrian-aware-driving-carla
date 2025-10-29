from typing import Dict, Iterable, Any

from .base_sensor import BaseSensor


class SensorManager:
    """
    Manage and coordinate multiple sensors in a CARLA simulation.

    This class maintains a dictionary of named sensors (subclasses of BaseSensor),
    automatically starts them listening upon initialization, and provides
    a unified interface to retrieve their processed data.
    """

    def __init__(self, sensors: Dict[str, BaseSensor]):
        """
        Initialize a SensorManager with a set of sensors.

        Parameters
        ----------
        sensors : Dict[str, BaseSensor]
            A dictionary where:
                - Key (str): The unique name for the sensor.
                - Value (BaseSensor): An instance of a BaseSensor subclass
                  (e.g., Camera, Lidar) already configured for attachment.

        Returns
        -------
        None
        """
        self._sensors = sensors

        # Start listening on all sensors as soon as they are registered
        for sensor in self._sensors.values():
            sensor.setup_listen()

    def get_data(self, sensor_names: Iterable[str], timeout: float = 10.0) -> Dict[str, Any]:
        """
        Retrieve processed data from specified sensors.

        Parameters
        ----------
        sensor_names : Iterable[str]
            An iterable of sensor names to fetch data from.
            Each name must correspond to a key in the `sensors` dictionary.
        timeout : float, optional
            Maximum time in seconds to wait for each sensor's data
            before raising a timeout exception. Default is 10.0 seconds.

        Returns
        -------
        Dict[str, Any]
            A dictionary where:
                - Key (str): The sensor name.
                - Value (Any): The processed data returned by the sensor’s
                  `get_data` method. The data type depends on the sensor:
                    * `PIL.Image.Image` for camera sensors.
                    * `Dict[str, np.ndarray]` for LiDAR sensors.
                    * Other sensor types may return custom formats.
        """
        return {name: self._sensors[name].get_data(timeout=timeout) for name in sensor_names}
