from typing import Dict, Iterable, Callable, Any
from functools import wraps
from inspect import signature


def requires_motion_fields(**param_field_map: Dict[str, Iterable[str]]) -> Callable:
    """
    Decorator to assert that certain parameters (expected to be motion‑estimation‑like objects)
    have required non‑None fields before calling the wrapped method.

    Parameters
    ----------
    **param_field_map : Dict[str, Iterable[str]]
        Mapping from parameter names to an iterable of field names that must be non‑None.

    Returns
    -------
    Callable
        The decorated function which performs runtime validation.

    Raises
    ------
    ValueError
        If a required parameter is missing or any required field on it is None.

    Examples
    --------
    >>> @requires_motion_fields(ego_motion=['position', 'velocity'], object_motion=['path'])
    ... def earliest_time_to_object_collision(self, ego_motion, object_motion):
    ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            sig = signature(func)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            for param, required_fields in param_field_map.items():
                input_data = bound.arguments.get(param)
                if input_data is None:
                    raise ValueError(
                        f"{func.__name__} missing required input parameter: '{param}'"
                    )
                for field in required_fields:
                    value = getattr(input_data, field, None)
                    if value is None:
                        raise ValueError(
                            f"{self.__class__.__name__}: Parameter '{param}' requires non‑None field: '{field}'"
                        )
            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator


def produces_motion_fields(*field_names: str) -> Callable:
    """
    Decorator to assert that the returned object from a function has certain non‑None fields.

    Parameters
    ----------
    *field_names : str
        Names of attributes that must be present and non‑None on the returned object.

    Returns
    -------
    Callable
        The decorated function which performs runtime validation.

    Raises
    ------
    ValueError
        If the returned object is missing any of the required fields.

    Examples
    --------
    >>> @produces_motion_fields('path', 'path_times')
    ... def estimate_motion(self, input_data):
    ...     # return a Motion with these fields populated
    ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, input_data: Any) -> Any:
            output = func(self, input_data)
            for field in field_names:
                value = getattr(output, field, None)
                if value is None:
                    raise ValueError(
                        f"{self.__class__.__name__} must produce non‑None output field: '{field}'"
                    )
            return output
        return wrapper
    return decorator
