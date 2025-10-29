from typing import Iterable, Optional, Dict

import carla
import numpy as np
from PIL import Image

from .utils import lidar_points_to_world, lidar_world_to_camera, lidar_camera_to_image


def camera_data_getter(image: carla.Image) -> Image:
    """
    Convert raw CARLA camera image data into a PIL Image (RGB).

    Parameters
    ----------
    image : carla.Image
        The raw image produced by a CARLA camera sensor.

    Returns
    -------
    PIL.Image.Image
        An RGB image constructed from the sensor's BGRA raw data.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA → RGB
    pil_image = Image.fromarray(rgb_array)
    return pil_image


def lidar_data_getter(
    point_cloud: carla.LidarMeasurement,
    ret_coord_spaces: Iterable[str],
    lidar_2_world: Optional[np.ndarray] = None,
    world_2_camera: Optional[np.ndarray] = None,
    image_w: Optional[int] = None,
    image_h: Optional[int] = None,
    fov: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Process a LiDAR point cloud into one or more coordinate spaces,
    with optional projection into image space.

    Parameters
    ----------
    point_cloud : carla.LidarMeasurement
        The raw LiDAR measurement containing 3D points and intensity values.
    ret_coord_spaces : Iterable[str]
        Collection of coordinate spaces to return. Options:
            'vehicle' : Points in LiDAR's local frame.
            'world'   : Points in world coordinates.
            'camera'  : Points in camera coordinates.
            'image'   : Points projected into 2D image coordinates.
    lidar_2_world : np.ndarray, optional
        4×4 transformation matrix from LiDAR sensor space to world space.
        Required if 'world', 'camera', or 'image' is requested.
    world_2_camera : np.ndarray, optional
        4×4 transformation matrix from world space to camera space.
        Required if 'camera' or 'image' is requested.
    image_w : int, optional
        Image width in pixels for projection. Required if 'image' is requested.
    image_h : int, optional
        Image height in pixels for projection. Required if 'image' is requested.
    fov : float, optional
        Horizontal field of view in degrees for projection. Required if 'image' is requested.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing some or all of the following keys, depending
        on `ret_coord_spaces`:
            'intensity' : np.ndarray of shape (N,)
                Intensity values for each point.
            'vehicle' : np.ndarray of shape (N, 3)
                Points in sensor-local coordinates.
            'world' : np.ndarray of shape (N, 4)
                Homogeneous points in world coordinates.
            'camera' : np.ndarray of shape (3, N)
                Points in camera coordinates.
            'image' : np.ndarray of shape (M, 3)
                2D image coordinates with depth; only points inside image bounds.

    Notes
    -----
    - When 'image' is requested, only visible points (within camera frustum
      and image bounds) are returned. All other arrays are filtered to match
      this visibility mask.
    - Ensure transformation matrices and camera parameters are provided if
      requesting 'world', 'camera', or 'image'.
    """
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape((-1, 4))
    points = data[:, :3]
    intensity = data[:, 3]

    result: Dict[str, np.ndarray] = {}
    result['intensity'] = intensity

    # Always compute vehicle space
    if 'vehicle' in ret_coord_spaces:
        result['vehicle'] = points

    world_points = None
    camera_points = None

    if {'world', 'camera', 'image'}.intersection(ret_coord_spaces):
        world_points = lidar_points_to_world(points, lidar_2_world)

    if {'camera', 'image'}.intersection(ret_coord_spaces):
        camera_points = lidar_world_to_camera(world_points, world_2_camera)

    if 'world' in ret_coord_spaces:
        result['world'] = world_points

    if 'camera' in ret_coord_spaces:
        result['camera'] = camera_points

    if 'image' in ret_coord_spaces:
        image_points, mask = lidar_camera_to_image(camera_points, image_w, image_h, fov)
        result['image'] = image_points
        result = {k: v[mask] for k, v in result.items()}

    return result
