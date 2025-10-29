from abc import ABC, abstractmethod
from queue import Queue
from typing import Callable, Any

import carla


class BaseSensor(ABC):
    """
    Abstract base class for all CARLA sensors.

    A BaseSensor:
    - Spawns a CARLA sensor actor and attaches it to the ego vehicle.
    - Listens to sensor output and stores raw data in an internal queue.
    - Processes data using a user‑provided `data_getter` function when retrieved.

    Subclasses must implement `get_data()` to define how data is retrieved
    and optionally processed further.
    """

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        bp: carla.ActorBlueprint,
        transform: carla.Transform,
        data_getter: Callable[[Any], Any]
    ):
        """
        Initialize a BaseSensor and spawn the sensor actor.

        Parameters
        ----------
        world : carla.World
            The CARLA world instance where the sensor will be spawned.
        ego_vehicle : carla.Vehicle
            The vehicle actor to which this sensor will be attached.
        bp : carla.ActorBlueprint
            The blueprint from which to spawn the sensor actor.
        transform : carla.Transform
            Transform specifying the sensor's position and orientation
            relative to the ego vehicle.
        data_getter : Callable[[Any], Any]
            A callable that takes raw sensor data from the queue and returns
            processed data.

        Returns
        -------
        None
        """
        self._bp = bp
        self._data_getter = data_getter
        self._queue = Queue()
        self._sensor = world.spawn_actor(bp, transform, attach_to=ego_vehicle)

    def get_bp(self) -> carla.ActorBlueprint:
        """
        Get the blueprint used to create this sensor.

        Returns
        -------
        carla.ActorBlueprint
            The blueprint associated with this sensor.
        """
        return self._bp

    def get_sensor(self) -> carla.Sensor:
        """
        Get the underlying CARLA sensor actor.

        Returns
        -------
        carla.Sensor
            The CARLA sensor actor instance.
        """
        return self._sensor

    def get_transform(self) -> carla.Transform:
        """
        Get the current global transform of the sensor.

        Returns
        -------
        carla.Transform
            The sensor's world transform.
        """
        return self._sensor.get_transform()

    def setup_listen(self) -> None:
        """
        Start listening to sensor output.

        This method registers an internal listener that places all received
        raw data into the internal queue.

        Returns
        -------
        None
        """
        self._sensor.listen(lambda data: self._queue.put(data))

    @abstractmethod
    def get_data(self, timeout: float = 10.0) -> Any:
        """
        Retrieve the next available processed data from the queue.

        Parameters
        ----------
        timeout : float, optional
            Maximum time in seconds to wait for data before raising
            `queue.Empty`. Default is 10.0.

        Returns
        -------
        Any
            The processed sensor data as returned by the `data_getter` function.
        """
        return self._data_getter(self._queue.get(timeout=timeout))
