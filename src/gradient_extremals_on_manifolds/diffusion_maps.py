"""Diffusion map coordinates module."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import scipy
import scipy.spatial
import scipy.sparse.linalg

import jax.numpy as jnp

from .procrustes import compute_msd_distance_matrix
#from gaussian_process import EuclideanGaussianProcess, RMSDGaussianProcess
from .utils import guess_epsilon
from .distances import EuclideanDistance, Distance


class DiffusionMaps():
    """Diffusion maps."""

    num_components: int         # Number of dimensions.
    epsilon: Optional[float]    # Spatial scale.
    points: jnp.ndarray         # Original points.
    mapped_points: jnp.ndarray  # Diffusion map coordinates.
    distance: Distance          # Distance metric to use
    _kernel_matrix: np.ndarray
    _eigenvalues: np.ndarray
    _eigenvectors: np.ndarray 

    def __init__(self, num_components: int,
                 epsilon: Optional[float] = None,
                 distance: Optional[Distance] = EuclideanDistance) -> None:
        self.num_components = num_components
        self.epsilon = epsilon
        self.d = distance

    def learn(self, points: jnp.ndarray) -> jnp.ndarray:
        """Learn diffusion map coordinates from points."""
        print("diffusion maps are learning")
        self.points = points
        squared_distance_matrix = self.d._compute_squared_distance_matrix(points)
        self.epsilon = guess_epsilon(squared_distance_matrix, self.epsilon)

        print("guessed epsilon.")
        self._kernel_matrix = np.exp(
            -squared_distance_matrix / (2.0 * self.epsilon**2))

        sqrt_diag_vector = np.sqrt(np.sum(self._kernel_matrix, axis=0))
        normalized_kernel_matrix = (
            (self._kernel_matrix / sqrt_diag_vector).T / sqrt_diag_vector).T

        print("diffusion maps are about to decompose")
        print(normalized_kernel_matrix.shape)
        ew, ev = scipy.sparse.linalg.eigsh(
            normalized_kernel_matrix, k=1 + self.num_components,
            v0=np.ones(normalized_kernel_matrix.shape[0]))
        indices = np.argsort(np.abs(ew))[::-1]
        ew = ew[indices]
        ev = ev[:, indices] / sqrt_diag_vector[:, None]
        sign = jnp.sign(jnp.sum(ev[:, 0]))
        print("eigenvalues: ", ew)
        print(ev.shape)
        self._eigenvalues = ew[1:]
        self._eigenvectors = ev[:,1:]#(ev / jnp.linalg.norm(ev, axis=0))[:,1:]
        self.mapped_points = sign * ev[:, 1:] * ew[1:]

        print("diffusion maps learned.")
        return self.mapped_points
