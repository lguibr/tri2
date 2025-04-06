import numpy as np
from typing import Tuple, Union, Dict, Any

# Adapted from Stable Baselines3 VecNormalize
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py


class RunningMeanStd:
    """Tracks the mean, variance, and count of values using Welford's algorithm."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream.
        :param epsilon: helps with arithmetic issues.
        :param shape: the shape of the data stream's output.
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """Creates a copy of the RunningMeanStd object."""
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = self.count
        return new_object

    def reset(self, epsilon: float = 1e-4) -> None:
        """Reset the statistics"""
        self.mean = np.zeros_like(self.mean)
        self.var = np.ones_like(self.var)
        self.count = epsilon

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        """
        Update the running mean and variance from a batch of samples.
        :param arr: Numpy array of shape (batch_size,) + shape
        """
        if arr.shape[1:] != self.mean.shape:
            raise ValueError(
                f"Expected input shape {self.mean.shape} (excluding batch dimension), got {arr.shape[1:]}"
            )

        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: Union[int, float],
    ) -> None:
        """
        Update the running mean and variance from batch moments.
        :param batch_mean: the mean of the batch
        :param batch_var: the variance of the batch
        :param batch_count: the number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        # Combine variances using Welford's method component analysis
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        # M2 = Combined sum of squares of differences from the mean
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Normalize an array using the running mean and variance."""
        # Ensure input is float64 for stability if needed, though usually input is float32
        # arr_float64 = arr.astype(np.float64)
        return (arr - self.mean) / np.sqrt(self.var + epsilon)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the running mean std for saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the running mean std from a saved dictionary."""
        self.mean = state_dict["mean"].copy()
        self.var = state_dict["var"].copy()
        self.count = state_dict["count"]
