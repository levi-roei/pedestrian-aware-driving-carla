import argparse
import sys

from simulation import default_agent_setup, SimulationManager


def main() -> None:
    """
    Launch the CARLA simulation with configurable parameters.

    This function parses command‑line arguments to configure and start
    a CARLA simulation. It sets up the simulation environment, adjusts
    Python's path for CARLA's API, instantiates a `SimulationManager`,
    and runs the simulation until completion. Once finished, the simulation
    is properly cleaned up.

    Parameters
    ----------
    None

    Command-line Arguments
    ----------------------
    host : str
        IP address of the CARLA host server.
    port : int
        Port for connecting to the CARLA server.
    tm_port : int
        Port for connecting to the Traffic Manager.
    is_asynchronous : bool
        Whether to run the simulation in asynchronous mode.
    fixed_delta_seconds : float
        Simulation step size in seconds (only applies when synchronous).
    timeout : float
        Timeout in seconds for client connections.
    seed : int
        Random seed for reproducibility.
    n_walkers : int
        Number of pedestrian agents to spawn.
    walker_max_speed : float
        Maximum speed for pedestrians in meters per second.
    carla_api_path : str
        Path to the CARLA Python API installation (used for carla agents).
    yolo_weights_path : str
        Path to the YOLO model weights used for object detection.
    deepsort_enc_weights_path : str
        Path to the DeepSort encoder weights used for tracking.
    pedestrian_collision_threshold : float
        Maximum distance in meters to register as a collision.
    pedestrian_slowdown_threshold : float
        Distance in meters for the slowdown safety margin.
    pedestrian_safety_threshold : float
        Minimum distance in meters from pedestrians considered safe.
    collision_horizon_dist : float
        Distance in meters over which collision prediction is performed.
    target_speed : float
        The target speed of the ego vehicle in km/h.

    Returns
    -------
    None
        The simulation runs until completion and logs details to stdout.
    """
    parser = argparse.ArgumentParser(description="CARLA Simulation Launcher")

    parser.add_argument('--host', type=str, default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                        help='Port to connect to the simulator (default: 2000)')
    parser.add_argument('--tm-port', type=int, default=8000,
                        help='Traffic Manager port (default: 8000)')
    parser.add_argument('--is-asynchronous', action='store_true',
                        help='Run simulation in asynchronous mode (default: False)')
    parser.add_argument('--fixed-delta-seconds', type=float, default=0.05,
                        help="Amount of simulation time per tick (default: 0.05 seconds). "
                             "Only applies when is-asynchronous is false (default).")
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Timeout for client connections (default: 10.0 seconds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-walkers', type=int, default=100,
                        help='Number of walkers to spawn (default: 100)')
    parser.add_argument('--walker-max-speed', type=float, default=1.4,
                        help='Maximum speed for walkers in m/s (default: 1.4)')
    parser.add_argument('--carla-api-path', type=str, default=r"D:\Carla\PythonAPI\carla",
                        help='Path to the CARLA Python API (default: D:\\Carla\\PythonAPI\\carla)')
    parser.add_argument('--yolo-weights-path', type=str, default='perception\\object_detection\\yolo12n.pt',
                        help='Path to the weights used by YOLO (default: perception\\object_detection\\yolo12n.pt)')
    parser.add_argument('--deepsort-enc-weights-path', type=str, default='perception\\object_tracking\\mars-small128.pb',
                        help='Path to the weights used by DeepSort encoder (default: perception\\object_tracking\\mars-small128.pb)')
    parser.add_argument('--pedestrian-collision-threshold', type=float, default=0.25,
                        help='Maximum distance in meters to be considered a collision. '
                             'Also the safety threshold for collision predictor. (default: 0.25 meters)')
    parser.add_argument('--pedestrian-slowdown-threshold', type=float, default=6.0,
                        help='The safety threshold for the slowdown collision predictor (default: 6.0 meters)')
    parser.add_argument('--pedestrian-safety-threshold', type=float, default=1.0,
                        help='Minimum distance in meters to be considered safely avoiding a pedestrian (default: 1.0 meters)')
    parser.add_argument('--collision-horizon-dist', type=float, default=50.0,
                        help='Distance over which to predict collisions (default: 60 meters)')
    parser.add_argument('--target-speed', type=float, default=20.0,
                        help='The target speed of the ego vehicle in km/h (default 20 km/h)')
    args = parser.parse_args()

    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"TM Port: {args.tm_port}")
    print(f"Asynchronous mode: {args.is_asynchronous}")
    print(f"Fixed delta seconds: {args.fixed_delta_seconds}")
    print(f"Timeout: {args.timeout}")
    print(f"Seed: {args.seed}")
    print(f"Number of walkers: {args.n_walkers}")
    print(f"Walker max speed: {args.walker_max_speed}")
    print(f"CARLA API path: {args.carla_api_path}")
    print(f"YOLO weights path: {args.yolo_weights_path}")
    print(f"DeepSort encoder weights path: {args.deepsort_enc_weights_path}")
    print(f"Pedestrian collision threshold: {args.pedestrian_collision_threshold} meters")
    print(f"Pedestrian slowdown threshold: {args.pedestrian_slowdown_threshold} meters")
    print(f"Pedestrian safety threshold: {args.pedestrian_safety_threshold} meters")
    print(f"Collision horizon distance: {args.collision_horizon_dist} meters")
    print(f"Target speed: {args.target_speed} km/h")

    # Extend Python path so Carla API can be imported
    sys.path.append(args.carla_api_path)

    simulation_manager = SimulationManager(args, default_agent_setup)
    simulation_manager.run()
    simulation_manager.destroy()


if __name__ == '__main__':
    main()
    print("\n\nFinished")
