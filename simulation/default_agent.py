import random
import argparse
from typing import List

import carla

from sensors import Camera, Lidar, SensorManager
from perception.object_detection import YOLODetector
from perception.object_tracking import DeepSortTracker
from objects_motion_estimation import LidarDetectorTrackerMotionEstimator
from motion_estimation import KFMotionEstimator
from decision_making.collision_prediction import ConstantSpeedCollisionPredictor
from decision_making import CollisionAwareDecisionMaker


def default_sensor_manager_setup(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    image_size_x: int = 800,
    image_size_y: int = 600,
    fov: float = 90.0,
    camera_transform: carla.Transform = carla.Transform(carla.Location(x=1.5, z=2.4)),
    lidar_range: float = 50.0,
    lidar_transform: carla.Transform = carla.Transform(carla.Location(x=0, z=2.5))
) -> SensorManager:
    """
    Set up and attach a default RGB camera and LiDAR sensor to the ego vehicle.

    Parameters
    ----------
    world : carla.World
        The CARLA world in which to spawn the sensors.
    ego_vehicle : carla.Vehicle
        The vehicle actor to which the sensors will be attached.
    image_size_x : int, optional
        Width of the camera image in pixels. Default is 800.
    image_size_y : int, optional
        Height of the camera image in pixels. Default is 600.
    fov : float, optional
        Horizontal field of view (degrees) for the camera. Default is 90.
    camera_transform : carla.Transform, optional
        Transform of the camera relative to the ego vehicle.
        Default is Transform at (x=1.5, z=2.4).
    lidar_range : float, optional
        Maximum LiDAR sensing range in meters. Default is 50.
    lidar_transform : carla.Transform, optional
        Transform of the LiDAR relative to the ego vehicle.
        Default is Transform at (x=0, z=2.5).

    Returns
    -------
    SensorManager
        A SensorManager instance containing the created camera and LiDAR.
    """
    bp_library = world.get_blueprint_library()
    
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_size_x))
    camera_bp.set_attribute('image_size_y', str(image_size_y))
    camera_bp.set_attribute('fov', str(fov))
    camera = Camera(world, ego_vehicle, camera_bp, camera_transform)

    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(lidar_range))
    lidar = Lidar(
        world,
        ego_vehicle,
        lidar_bp,
        lidar_transform,
        ret_coordinate_spaces=['vehicle', 'world', 'camera', 'image'],
        camera_to_project_to=camera
    )

    sensor_manager = SensorManager({'camera': camera, 'lidar': lidar})
    return sensor_manager


def default_agent_setup(
    ego_vehicle: carla.Vehicle,
    world: carla.World,
    args: argparse.Namespace,
    detector_conf_threshold: float = 0.0,
    detector_classes: List[str] = ['person'],
    **sensor_manager_kwargs
) -> 'CollisionAwareAgent':  # Cannot directly import CollisionAwareAgent due to order
    """
    Set up a default collision‑aware agent with object detection and tracking.

    This function:
    - Selects a random destination from available spawn points.
    - Builds a SensorManager with a camera and LiDAR.
    - Initializes YOLO detection and DeepSort tracking.
    - Creates a motion estimator combining LiDAR detections and tracked objects.
    - Configures a CollisionAwareAgent with collision prediction modules.

    Parameters
    ----------
    ego_vehicle : carla.Vehicle
        The ego vehicle actor that the agent will control.
    world : carla.World
        The CARLA world in which the simulation runs.
    args : argparse.Namespace
        Parsed arguments for configuration. Expected attributes include:
            collision_horizon_dist : float
                Horizon distance in meters for collision prediction.
            yolo_weights_path : str
                Path to the YOLO model weights.
            deepsort_enc_weights_path : str
                Path to the DeepSort encoder weights.
            pedestrian_collision_threshold : float
                Distance threshold in meters for collisions.
            pedestrian_slowdown_threshold : float
                Distance threshold in meters for slowdown prediction.
    detector_conf_threshold : float, optional
        Confidence threshold for object detection. Default is 0.0.
    detector_classes : List[str], optional
        List of classes to detect (e.g., ['person']). Default is ['person'].
    sensor_manager_kwargs:
        Keyword arguments to set up the sensor manager.

    Returns
    -------
    CollisionAwareAgent
        A fully configured collision‑aware agent ready to be executed.
    """
    # Select a random destination
    vehicle_sps = world.get_map().get_spawn_points()
    agent_dest = random.choice(vehicle_sps).location
    world.debug.draw_string(agent_dest, "destination", life_time=1000000)

    # Build sensor manager and motion estimator
    sensor_manager = default_sensor_manager_setup(world, ego_vehicle, **sensor_manager_kwargs)
    sensor_config = {'image': 'camera', 'lidar': 'lidar'}
    object_detector = YOLODetector(
        args.yolo_weights_path,
        confidence_threshold=detector_conf_threshold,
        classes=detector_classes
    )
    object_tracker = DeepSortTracker(args.deepsort_enc_weights_path)
    objects_motion_estimator = LidarDetectorTrackerMotionEstimator(
        sensor_manager,
        sensor_config,
        object_detector,
        object_tracker,
        KFMotionEstimator
    )

    # Import agents only here to avoid problems with the path of agents.navigation.basic_agent
    # which depends on the location of the CARLA folder and thus provided using a CLI argument 
    from agents.navigation.basic_agent import BasicAgent
    from collision_aware_agent import CollisionAwareAgent

    collision_predictor = ConstantSpeedCollisionPredictor(
        safety_threshold=args.pedestrian_collision_threshold
    )
    collision_predictor_slowdown = ConstantSpeedCollisionPredictor(
        safety_threshold=args.pedestrian_slowdown_threshold
    )
    decision_maker = CollisionAwareDecisionMaker(
        args.target_speed,
        collision_predictor,
        collision_predictor_slowdown,
        args.collision_horizon_dist
    )

    basic_agent = BasicAgent(ego_vehicle)
    basic_agent.ignore_traffic_lights()  # For testing: ignore traffic lights

    coll_aware_agent = CollisionAwareAgent(
        ego_vehicle,
        world,
        basic_agent,
        objects_motion_estimator,
        decision_maker
    )
    coll_aware_agent.set_destination(agent_dest)
    return coll_aware_agent
