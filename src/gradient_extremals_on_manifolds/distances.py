

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp
import jax

from .procrustes import msd, compute_msd_distance_matrix

from scipy.spatial import distance

"""  Data is of form [no_samples:no_atoms:no_dims]"""

class Distance(ABC):
    """Distance base class."""

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _squared_distance(x: jnp.array, y: jnp.array) -> float:
        ...

    @staticmethod
    @abstractmethod
    def _compute_squared_distance_matrix(points: jnp.ndarray) \
            -> jnp.ndarray:
        ...

class EuclideanDistance(Distance):
    """Standard Euclidean Distance Metric."""

    @staticmethod
    def _squared_distance(x: jnp.array, y: jnp.array) -> float:
        v = (x - y).flatten()
        return jnp.dot(v, v)

    @staticmethod
    def _compute_squared_distance_matrix(points: jnp.ndarray) \
            -> jnp.ndarray:
        return distance.squareform(
            distance.pdist(points.reshape(points.shape[0], -1),
                           metric='sqeuclidean'))

class RMSDDistance(Distance):
    """RMSD distance."""

    def __init__(self, no_atoms) -> None:
        self.no_atoms = no_atoms
    
    @staticmethod
    def _squared_distance(x: jnp.array, y: jnp.array) -> float:
        return msd(x, y)

    @staticmethod
    def _compute_squared_distance_matrix(points: jnp.ndarray) \
            -> jnp.ndarray:
        return compute_msd_distance_matrix(points)
