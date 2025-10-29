from typing import Optional, Dict, Any, List, Callable
import random
import argparse

import carla

from .utils import move_spectator


class SimulationManager:
    """
    Manage the CARLA simulation lifecycle, including world setup, actor spawning,
    agent control, and statistics collection.

    This class is responsible for:
    - Connecting to the CARLA server and configuring the world.
    - Spawning the ego vehicle and pedestrian actors.
    - Running the provided agent until completion.
    - Tracking safety-related statistics and cleaning up after simulation.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        setup_agent_function: Callable[..., 'BaseCollisionAwareAgent'],
        # Cannot directly import BaseCollisionAwareAgent due to order
        **setup_agent_kwargs
    ):
        """
        Initialize the SimulationManager with agent setup and parsed arguments.

        Parameters
        ----------
        setup_agent_function : Callable[..., BaseCollisionAwareAgent]
            A function that accepts the ego vehicle, world, args, and optional keyword arguments returning
            a BaseCollisionAwareAgent.
        args : argparse.Namespace
            Parsed simulation configuration, including:
                host : str
                port : int
                tm_port : int
                is_asynchronous : bool
                fixed_delta_seconds : float
                timeout : Optional[int]
                seed : Optional[int]
                n_walkers : int
                walker_max_speed : float
                pedestrian_safety_threshold : float
                pedestrian_collision_threshold : float
        setup_agent_kwargs
            Optional keyword arguments for the setup agent function.
        """
        self._args = args

        world_dict = self.setup_world(
            self._args.host,
            self._args.port,
            self._args.tm_port,
            self._args.is_asynchronous,
            self._args.fixed_delta_seconds,
            self._args.timeout,
            self._args.seed
        )
        self._client = world_dict['client']
        self._world = world_dict['world']
        self._tm = world_dict['tm']

        self._ego_vehicle = self.setup_ego_vehicle(self._world)

        walkers_setup = self.setup_walkers(self._world, self._args.n_walkers, self._args.walker_max_speed)
        self._walkers = walkers_setup['walkers']
        self._controllers = walkers_setup['controllers']

        self._agent = setup_agent_function(self._ego_vehicle, self._world, self._args, **setup_agent_kwargs)

    def setup_world(
        self,
        host: str,
        port: int,
        tm_port: int,
        is_asynchronous: bool,
        fixed_delta_seconds: float,
        timeout: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Connect to the CARLA server and configure the world.

        Parameters
        ----------
        host : str
            Hostname or IP address of the CARLA server.
        port : int
            Port number for the CARLA server.
        tm_port : int
            Port number for the Traffic Manager.
        is_asynchronous : bool
            Whether to run in asynchronous mode.
        fixed_delta_seconds : float
            Simulation step size in seconds for synchronous mode.
        timeout : Optional[int], default=None
            Connection timeout in seconds.
        seed : Optional[int], default=None
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
                'client' : carla.Client
                    Connected CARLA client instance.
                'world' : carla.World
                    Active world object.
                'tm' : carla.TrafficManager
                    Traffic manager instance.
        """
        client = carla.Client(host, port)
        if timeout:
            client.set_timeout(timeout)
        if seed:
            random.seed(seed)

        world = client.get_world()
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_random_device_seed(seed)

        if not is_asynchronous:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_delta_seconds
            world.apply_settings(settings)

        return {'client': client, 'world': world, 'tm': traffic_manager}

    def setup_ego_vehicle(self, world: carla.World) -> carla.Vehicle:
        """
        Spawn an ego vehicle in the world at a random spawn point.

        Parameters
        ----------
        world : carla.World
            The CARLA world where the vehicle will be spawned.

        Returns
        -------
        carla.Vehicle
            The spawned ego vehicle actor.
        """
        bp_library = world.get_blueprint_library()
        vehicle_sps = world.get_map().get_spawn_points()
        vehicle_bps = bp_library.filter('vehicle.*')
        vehicle_bp = random.choice(vehicle_bps)
        transform = random.choice(vehicle_sps)
        world.debug.draw_string(transform.location, "spawn", life_time=9999999999)
        ego_vehicle = world.spawn_actor(vehicle_bp, transform)
        world.tick()
        return ego_vehicle

    def setup_walkers(self,
                      world: carla.World,
                      n_walkers: int,
                      walker_max_speed: float) -> Dict[str, List[carla.Actor]]:
        """
        Spawn pedestrian actors (walkers) and their AI controllers.

        Parameters
        ----------
        world : carla.World
            The CARLA world where walkers will be spawned.
        n_walkers : int
            Number of walkers to spawn.
        walker_max_speed : float
            Maximum speed in meters per second for walkers.

        Returns
        -------
        Dict[str, List[Any]]
            Dictionary with keys:
                'walkers' : List[carla.Actor]
                    Spawned walker actors.
                'controllers' : List[carla.Actor]
                    Controller actors managing the walkers.
        """
        bp_library = world.get_blueprint_library()
        walker_bps = bp_library.filter('walker.pedestrian.*')
        walkers: List[Any] = []

        while len(walkers) < n_walkers:
            walker_sp = carla.Transform()
            walker_loc = world.get_random_location_from_navigation()
            walker_sp.location = walker_loc
            walker_bp = random.choice(walker_bps)
            walker = world.try_spawn_actor(walker_bp, walker_sp)
            if walker is not None:
                walkers.append(walker)

        walker_controller_bp = bp_library.find('controller.ai.walker')
        controllers: List[Any] = []
        for walker in walkers:
            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            controllers.append(controller)
        world.tick()

        for controller in controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(walker_max_speed)

        return {'walkers': walkers, 'controllers': controllers}

    def run(
        self,
        spectator_loc: carla.Location = carla.Location(z=20),
        spectator_rot: carla.Rotation = carla.Rotation(pitch=-90)
    ) -> None:
        """
        Run the main simulation loop until the agent signals completion.

        Parameters
        ----------
        spectator_loc : carla.Location, optional
            Position of the spectator camera relative to the ego vehicle.
            Defaults to Location(z=20).
        spectator_rot : carla.Rotation, optional
            Orientation of the spectator camera. Defaults to looking straight down.

        Returns
        -------
        None
            The simulation runs until completion and prints statistics.
        """
        spectator = self._world.get_spectator()

        n_collisions = 0
        n_unsafe_interactions = 0
        min_walker_dist = float('inf')

        print("\n\nStarted Simulation")

        while True:
            self._world.tick()

            if self._agent.done():
                break

            control = self._agent.run_step()
            self._ego_vehicle.apply_control(control)

            ego_loc = self._ego_vehicle.get_location()

            min_dist_walker_timestep = float('inf')
            for walker in self._walkers:
                walker_loc = walker.get_location()
                walker_dist = walker_loc.distance(ego_loc)
                if walker_dist < min_dist_walker_timestep:
                    min_dist_walker_timestep = walker_dist

            if min_dist_walker_timestep <= self._args.pedestrian_safety_threshold:
                n_unsafe_interactions += 1
                if min_dist_walker_timestep <= self._args.pedestrian_collision_threshold:
                    n_collisions += 1
                    print("Collision")
                else:
                    print("Unsafe interaction")

            if min_dist_walker_timestep < min_walker_dist:
                min_walker_dist = min_dist_walker_timestep

            speed_kmh = self._ego_vehicle.get_velocity().length() * 3.6
            print(f"Ego vehicle speed is {speed_kmh:.2f} km/h")

            vehicle_transform = self._ego_vehicle.get_transform()
            move_spectator(vehicle_transform, spectator, spectator_loc, spectator_rot)

        print(f"\n\nNumber of collisions: {n_collisions}")
        print(f"Number of unsafe interactions: {n_unsafe_interactions}")
        est_ttc_stats = self._agent.get_estimated_ttc_stats()
        ttc_mean, ttc_min = est_ttc_stats['mean'], est_ttc_stats['min']
        if ttc_mean is None:
            print("No collision was predicted during the simulation")
        else:
            print(f"When a collision was predicted, the average TTC was {ttc_mean:.2f} seconds")
            print(f"Minimum estimated TTC: {ttc_min:.2f} seconds")
        print(f"Run step latency avg: {self._agent.get_run_step_latency_avg():.2f} seconds")

    def destroy(self) -> None:
        """
        Clean up all spawned actors and restore world settings.

        Resets the world to asynchronous mode and destroys all walker
        actors and their controllers. Should be called after `run()`
        to prevent leaking actors or leaving the world in synchronous mode.

        Returns
        -------
        None
        """
        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)

        for controller in self._controllers:
            controller.stop()

        for walker, controller in zip(self._walkers, self._controllers):
            if controller.is_active:
                controller.destroy()
            if walker.is_active:
                walker.destroy()
