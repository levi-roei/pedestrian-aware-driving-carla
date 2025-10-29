import carla


def move_spectator(
    vehicle_transform: carla.Transform,
    spectator: carla.Actor,
    add_loc: carla.Location,
    rot: carla.Rotation
) -> None:
    """
    Move the spectator camera relative to the given vehicle.

    Parameters
    ----------
    vehicle_transform : carla.Transform
        The transform (location & rotation) of the reference vehicle.
    spectator : carla.Actor
        The spectator actor (camera) to move.
    add_loc : carla.Location
        Location offset to apply relative to the vehicle's position.
    rot : carla.Rotation
        Rotation to set for the spectator camera.

    Returns
    -------
    None
        This function updates the spectator's transform in place.
    """
    spectator_transform = carla.Transform(
        vehicle_transform.location + add_loc,
        rot
    )
    spectator.set_transform(spectator_transform)
