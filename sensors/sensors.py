from typing import List, Optional, Dict, Any

from PIL import Image
import carla
import numpy as np

from .base_sensor import BaseSensor
from sensors.data_getters import camera_data_getter, lidar_data_getter


class Camera(BaseSensor):
    def __init__(self,
                 world: carla.World,
                 ego_vehicle: carla.Vehicle,
                 bp: carla.ActorBlueprint,
                 transform: carla.Transform):
        """
        Camera sensor wrapper for CARLA.

        This class attaches a camera to the given ego vehicle, routes its
        raw CARLA data into an internal queue, and provides a method to
        retrieve processed image data via the `camera_data_getter`.

        Args:
            world (carla.World): The CARLA world instance.
            ego_vehicle (carla.Actor): The vehicle to which the camera is attached.
            bp (carla.ActorBlueprint): The blueprint of the camera sensor.
            transform (carla.Transform): Transform to attach the camera relative to the vehicle.
        """
        data_getter = camera_data_getter
        super().__init__(world, ego_vehicle, bp, transform, data_getter)


    def get_data(self, timeout: float = 10.0) -> Image:
        """
        Retrieve the next available frame from the camera.

        This method blocks until the next raw camera data is available in the
        internal queue, then processes it through `camera_data_getter`.

        Args:
            timeout (float): Seconds to wait for data before raising an exception.

        Returns:
            The processed image data (e.g., a PIL Image).
        """
        return super().get_data(timeout=timeout)


class Lidar(BaseSensor):
    def __init__(self,
                 world: carla.World,
                 ego_vehicle: carla.Vehicle,
                 bp: carla.ActorBlueprint,
                 transform: carla.Transform,
                 ret_coordinate_spaces: List[str] = ['vehicle'],
                 camera_to_project_to: Optional[Camera] = None):
        """
        LiDAR sensor wrapper for CARLA.

        This class attaches a LiDAR sensor to the given ego vehicle, routes its
        raw CARLA data into an internal queue, and provides a method to retrieve
        processed point cloud data via the `lidar_data_getter`.

        Args:
            world (carla.World): The CARLA world instance.
            ego_vehicle (carla.Actor): The vehicle to which the LiDAR sensor is attached.
            bp (carla.ActorBlueprint): The blueprint of the LiDAR sensor.
            transform (carla.Transform): Transform to attach the LiDAR relative to the vehicle.
            ret_coordinate_spaces (list of str): Desired coordinate spaces for the output.
                Must be a subset of ['vehicle', 'world', 'camera', 'image'].
            camera_to_project_to (Camera, optional): A Camera instance for projecting
                LiDAR points into camera/image space. Must be provided if
                'camera' or 'image' is included in `ret_coordinate_spaces`.
        """
        valid_spaces = {'vehicle', 'world', 'camera', 'image'}
        requested_spaces = set(ret_coordinate_spaces)

        # Validate coordinate spaces
        if not requested_spaces.issubset(valid_spaces):
            invalid = requested_spaces - valid_spaces
            raise ValueError(f"Invalid coordinate spaces requested: {invalid}. "
                             f"Allowed values are: {valid_spaces}")

        # Enforce camera projection logic consistency
        needs_camera = 'camera' in requested_spaces or 'image' in requested_spaces
        if needs_camera and camera_to_project_to is None:
            raise ValueError("camera_to_project_to must be provided when 'camera' or 'image' is in ret_coordinate_spaces.")
        if camera_to_project_to is not None and not needs_camera:
            raise ValueError("camera_to_project_to was provided, but 'camera' or 'image' was not requested in ret_coordinate_spaces.")

        self._ret_coordinate_spaces = ret_coordinate_spaces
        self._camera_to_project_to = camera_to_project_to
        data_getter = lidar_data_getter
        super().__init__(world, ego_vehicle, bp, transform, data_getter)


    def get_data(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Retrieve the next available frame from the LiDAR.

        This method blocks until the next raw LiDAR data is available in the
        internal queue, then processes it through `lidar_data_getter` with the
        specified coordinate spaces. If a camera is provided, the point cloud
        will also be projected into that camera's view.

        Args:
            timeout (float): Seconds to wait for data before raising an exception.

        Returns:
            dict: The processed LiDAR data, with keys depending on the
                  requested coordinate spaces. For example:
                  {
                    'vehicle': np.ndarray of points,
                    'world': np.ndarray of points,
                    'camera': np.ndarray of points,
                    'image': np.ndarray of image coordinates
                  }
        """
        if self._camera_to_project_to is not None:
            return self._data_getter(self._queue.get(timeout=timeout),
                                     self._ret_coordinate_spaces,
                                     lidar_2_world=self._sensor.get_transform().get_matrix(),
                                     world_2_camera=np.array(self._camera_to_project_to.get_transform().get_inverse_matrix()),
                                     image_w=self._camera_to_project_to.get_bp().get_attribute("image_size_x").as_int(),
                                     image_h=self._camera_to_project_to.get_bp().get_attribute("image_size_y").as_int(),
                                     fov=self._camera_to_project_to.get_bp().get_attribute("fov").as_float())
        else:
            return self._data_getter(self._queue.get(timeout=timeout),
                              self._ret_coordinate_spaces)
