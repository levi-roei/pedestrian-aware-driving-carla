import numpy as np


def lidar_points_to_world(points: np.ndarray, lidar_2_world: np.ndarray) -> np.ndarray:
    """
    Transform a LiDAR point cloud from the sensor frame to world coordinates.

    Parameters
    ----------
    points : np.ndarray
        Point cloud in LiDAR sensor space of shape (N, 3),
        where N is the number of points.
    lidar_2_world : np.ndarray
        4×4 transformation matrix from LiDAR to world coordinates.

    Returns
    -------
    np.ndarray
        Transformed points in world coordinates with shape (N, 3).
    """
    local_lidar_points = points.T
    local_lidar_points = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
    world_points = np.dot(lidar_2_world, local_lidar_points)
    return world_points[:3].T


def lidar_world_to_camera(world_points: np.ndarray, world_2_camera: np.ndarray) -> np.ndarray:
    """
    Transform points from world coordinates to the camera coordinate system.

    Parameters
    ----------
    world_points : np.ndarray
        Input points in world coordinates with shape (N, 3),
        where N is the number of points.
    world_2_camera : np.ndarray
        4×4 transformation matrix from world to camera.

    Returns
    -------
    np.ndarray
        Points in camera coordinates with shape (N, 3),
        following the standard camera convention used by OpenCV.
    """
    cloud_size = world_points.shape[0]
    world_points_T = world_points.T
    ones_row = np.ones((1, cloud_size))
    world_points_h = np.vstack([world_points_T, ones_row])

    sensor_points = np.dot(world_2_camera, world_points_h)

    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]
    ])
    return point_in_camera_coords.T


def lidar_camera_to_image(
    point_in_camera_coords: np.ndarray,
    image_w: int,
    image_h: int,
    fov: float
) -> (np.ndarray, np.ndarray):
    """
    Project camera-space 3D points into 2D image coordinates.

    Parameters
    ----------
    point_in_camera_coords : np.ndarray
        Points in camera coordinates of shape (N, 3),
        where N is the number of points.
    image_w : int
        Image width in pixels.
    image_h : int
        Image height in pixels.
    fov : float
        Horizontal field of view in degrees.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - points_2d : np.ndarray
            Projected 2D points with shape (N, 3),
            where columns are [x, y, depth].
        - points_in_canvas_mask : np.ndarray
            Boolean mask of shape (N,) indicating
            which points lie within the image bounds
            and are in front of the camera.
    """
    point_in_camera_coords = point_in_camera_coords.T
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0

    points_2d = np.dot(K, point_in_camera_coords)
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]
    ])

    points_2d = points_2d.T
    points_in_canvas_mask = (
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) &
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) &
        (points_2d[:, 2] > 0.0)
    )

    return points_2d, points_in_canvas_mask
