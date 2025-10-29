from typing import Optional, Dict
import time

import numpy as np
import carla
from agents.navigation.basic_agent import BasicAgent

from motion_estimation import Motion
from objects_motion_estimation import BaseObjectsMotionEstimator
from decision_making import BaseDecisionMaker
from utils import EWMA
from .base_collision_aware_agent import BaseCollisionAwareAgent


class CollisionAwareAgent(BaseCollisionAwareAgent):
    """
    A collision‑aware autonomous driving agent for CARLA.

    This agent wraps a CARLA `BasicAgent` and delegates high‑level
    decision‑making (speed adjustments and emergency stop decisions)
    to an injected `BaseDecisionMaker` implementation. It remains
    responsible for:
      • Gathering the ego vehicle's current motion state,
      • Passing the motion states of the ego vehcile and objects into the decision maker,
      • Applying the resulting target speed and emergency stop
        commands to the underlying `BasicAgent`.

    **Field Requirements**
    ----------------------
    When calling `decision_maker.make_decision(ego_motion, objects_motion)`:
    • The `ego_motion` provided contains `path` and `speed`.
      Any fields that the decision maker (or its internal collision predictors)
      require **must be a subset of** these two fields.
    • The `objects_motion` provided is exactly what the configured
      `BaseObjectsMotionEstimator` returns. Any fields required by the decision maker
      (or its internal predictors) **must be a subset of the fields produced by**
      that estimator.

    Attributes
    ----------
    _vehicle : carla.Vehicle
        The ego vehicle controlled by this agent.
    _world : carla.World
        The CARLA world, used to retrieve simulation time.
    _agent : BasicAgent
        The underlying CARLA agent used for navigation and low‑level control.
    _objects_motion_estimator : BaseObjectsMotionEstimator
        Module that predicts the future motion of surrounding objects.
    _decision_maker : BaseDecisionMaker
        Strategy object that decides target speed and whether to emergency stop.
    latency_ewma : EWMA
        Exponential weighted moving average filter tracking `run_step` latency.
    """

    def __init__(
        self,
        ego_vehicle: carla.Vehicle,
        world: carla.World,
        agent: BasicAgent,
        objects_motion_estimator: BaseObjectsMotionEstimator,
        decision_maker: BaseDecisionMaker,
        latency_ewma_alpha: float = 0.1
    ) -> None:
        """
        Initialize the collision‑aware agent.

        Parameters
        ----------
        ego_vehicle : carla.Vehicle
            The vehicle actor to control.
        world : carla.World
            The CARLA world, used to retrieve current simulation time.
        agent : BasicAgent
            A CARLA `BasicAgent` instance for path following and actuation.
        objects_motion_estimator : BaseObjectsMotionEstimator
            Component that estimates motion for nearby objects each step.
        decision_maker : BaseDecisionMaker
            Component responsible for high‑level decision making (target speed,
            emergency stop) based on ego/object motion inputs.
        latency_ewma_alpha : float, optional
            Smoothing factor for the latency EWMA. Defaults to 0.1.
        """
        self._vehicle = ego_vehicle
        self._world = world
        self._agent = agent
        self._objects_motion_estimator = objects_motion_estimator
        self._decision_maker = decision_maker
        self.latency_ewma = EWMA(latency_ewma_alpha)

        # Initialize agent target speed using decision maker’s default if available
        # (some decision makers may expose target_speed externally)
        if hasattr(self._decision_maker, "_target_speed"):
            self._agent.set_target_speed(self._decision_maker._target_speed)

    def run_step(self) -> carla.VehicleControl:
        """
        Execute one control step of the collision‑aware agent.

        Workflow:
        1. Estimate the motion of surrounding objects using `_objects_motion_estimator`.
        2. Package the ego vehicle's current path (from route plan) and speed.
        3. Delegate to `_decision_maker.make_decision()` with ego and objects motion.
        4. Apply the decision by updating the underlying `BasicAgent`:
           - Set new target speed,
           - Optionally apply an emergency stop.
        5. Update latency statistics and return the final `VehicleControl`.

        Returns
        -------
        carla.VehicleControl
            The control command to apply to the ego vehicle.
        """
        start = time.perf_counter()

        # Estimate motion of surrounding objects
        simulation_time = self._world.get_snapshot().timestamp.elapsed_seconds
        objects_motion = self._objects_motion_estimator.estimate_objects_motion(
            simulation_time=simulation_time
        )

        # Extract planned route and current ego speed
        agent_plan = self._agent.get_local_planner().get_plan()
        locations = [step[0].transform.location for step in agent_plan]
        path = [np.array([loc.x, loc.y, loc.z]) for loc in locations]

        ego_speed = self._vehicle.get_velocity().length()
        ego_motion = Motion(path=path, speed=ego_speed)

        # Delegate decision making
        decision = self._decision_maker.make_decision(ego_motion, objects_motion)
        target_speed = decision["target_speed"]
        emergency_stop = decision["emergency_stop"]

        # Apply decision to underlying agent
        self._agent.set_target_speed(target_speed)
        control = self._agent.run_step()
        if emergency_stop:
            control = self._agent.add_emergency_stop(control)

        # Track latency
        self.latency_ewma.update(time.perf_counter() - start)

        return control

    def set_destination(self, destination: carla.Location) -> None:
        """
        Set a new destination for the underlying BasicAgent.

        Parameters
        ----------
        destination : carla.Location
            The target location the agent should navigate to.
        """
        self._agent.set_destination(destination)

    def done(self) -> bool:
        """
        Check whether the agent has reached its destination.

        Returns
        -------
        bool
            True if the agent has arrived, False otherwise.
        """
        return self._agent.done()

    def get_estimated_ttc_stats(self) -> Dict[str, Optional[float]]:
        """
        Retrieve statistics on the estimated time‑to‑collision (TTC)
        observed so far.

        Returns
        -------
        Dict[str, Optional[float]]
            A dictionary containing TTC metrics, for example:
                "mean" : Optional[float]
                    The smoothed or average TTC value in seconds,
                    or None if no collisions have been predicted yet.
                "min" : Optional[float]
                    The smallest TTC value observed (in seconds),
                    or None/float('inf') if no collision has been predicted.
        """
        return self._decision_maker.get_estimated_ttc_stats()

    def get_run_step_latency_avg(self) -> Optional[float]:
        """
        Get the smoothed average latency of recent `run_step` calls.

        Returns
        -------
        Optional[float]
            The EWMA of recent latencies, or None if no data yet.
        """
        return self.latency_ewma.get_avg()
