from typing import Optional


class EWMA:
    """
    Exponentially Weighted Moving Average (EWMA) filter.

    Parameters
    ----------
    alpha : float, optional
        Smoothing factor in [0, 1].
        Lower alpha => smoother, less reactive.
        Higher alpha => more reactive to recent changes.
        Default is 0.2.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha: float = alpha
        self._ewma: Optional[float] = None

    def update(self, new_val: float) -> float:
        """
        Update the EWMA with a new measurement.

        Parameters
        ----------
        new_val : float
            The new measurement to incorporate.

        Returns
        -------
        float
            The updated EWMA value.
        """
        if self._ewma is None:
            # Initialize with the first value
            self._ewma = new_val
        else:
            self._ewma = self._alpha * new_val + (1 - self._alpha) * self._ewma
        return self._ewma

    def get_avg(self) -> Optional[float]:
        """
        Get the current EWMA value.

        Returns
        -------
        Optional[float]
            The current EWMA, or None if no value has been set yet.
        """
        return self._ewma
