"""
Gaussian process regression.
Code forked from Juan Bello-Rivas
https://github.com/jmbr/staying-the-course-code/blob/main/
        src/staying_the_course/gaussian_process.py
"""

__all__ = ['GaussianProcess', 'make_gaussian_process']

from typing import Optional
from functools import partial

from .distances import EuclideanDistance, Distance
from .utils import guess_epsilon

from scipy.spatial import distance

import jax
import jax.scipy
import jax.numpy as jnp


DEFAULT_SIGMA: float = 1e-2 #1e-2


class GaussianProcess:
    """Gaussian process regressor."""

    sigma: float = DEFAULT_SIGMA  # Regularization term.
    epsilon: Optional[float] = None  # Spatial scale parameter.
    alphas: Optional[jnp.ndarray] = None  # Coefficients.
    points: jnp.ndarray
    cholesky_factor: jnp.ndarray
    distance: Distance          # Distance metric to use


    def __init__(
        self, sigma: Optional[float] = None, epsilon: Optional[float] = None,
        distance: Optional[Distance] = EuclideanDistance
    ) -> None:
        if sigma is not None:
            self.sigma = sigma
        if epsilon is not None:
            self.epsilon = epsilon
        self.d = distance

    def _learn(
        self,
        points: jnp.ndarray,
        values: jnp.ndarray,
        epsilon: float,
        kernel_matrix: jnp.ndarray,
    ) -> None:
        """Auxiliary method for fitting a Gaussian process."""
        self.points = points
        self.epsilon = epsilon

        sigma2_eye = self.sigma**2 * jnp.eye(kernel_matrix.shape[0])
        L, _ = jax.scipy.linalg.cho_factor(
            kernel_matrix + sigma2_eye, lower=True, check_finite=False
        )
        self.cholesky_factor = L
        self.alphas = jax.scipy.linalg.cho_solve(
            (L, True), values, check_finite=False
        )

    def learn(self, points: jnp.ndarray, values: jnp.ndarray) -> None:
        """Fit a Gaussian process

        Parameters
        ----------
        points: jnp.ndarray
            Data points arranged by rows.
        values: jnp.ndarray
            Values corresponding to the data points. These can be scalars or
            arrays (arranged by rows).

        """
        threshold = jnp.finfo(points.dtype).eps * 1e2
        squared_distance_matrix = self.d._compute_squared_distance_matrix(points)
        self.epsilon = guess_epsilon(squared_distance_matrix, self.epsilon, threshold=threshold)
        print("EPSILON: ", self.epsilon)

        kernel_matrix = jnp.exp(
            -squared_distance_matrix / (2.0 * self.epsilon**2))

        diagonal_indices = jnp.diag_indices_from(kernel_matrix)
        kernel_matrix = kernel_matrix.at[diagonal_indices].set(1.0)
        self.kernel_matrix = kernel_matrix

        self._learn(points, values, self.epsilon, self.kernel_matrix)

    @partial(jax.jit, static_argnums=0)
    def __call__(self, point: jnp.ndarray) -> jnp.ndarray:
        """Evaluate Gaussian process at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: jnp.ndarray
            A single point on which the previously learned Gaussian process
            is to be evaluated.

        Returns
        -------
        value: jnp.ndarray
            Estimated value of the GP at the given point.

        """
        d2 = self.d._squared_distance
        def kernel(x):
            return jnp.exp(-d2(x, point) / (2.0 * self.epsilon**2))

        kstar = jax.vmap(kernel)(self.points)

        # print("LOOK: ", kstar.shape)
        return kstar @ self.alphas

    @partial(jax.jit, static_argnums=0)
    def get_variance(self, point: jnp.ndarray) -> jnp.ndarray:
        """Evaluate variance of GP at a new point.
        This function must be called after the Gaussian process has been
        fitted using the `learn` method.
        Parameters
        ----------
        point: jnp.ndarray
            A single point on which the previously learned Gaussian process
            is to be evaluated.
        Returns
        -------
        value: jnp.ndarray
            Estimated value of the GP variance at the given point.
        """
        d2 = self.d._squared_distance
        def kernel(x):
            return jnp.exp(-d2(x, point) / (2.0 * self.epsilon**2))

        kstar = jax.vmap(kernel)(self.points)

        betas = jax.scipy.linalg.cho_solve((self.cholesky_factor, True),
                                           kstar, check_finite=False)
        return 1 - kstar @ betas

    def save_gpr(self, path_to_folder):
        """Saves out necessary information to evaluate gpr as numpy arrays. 
        Can be used to transfer gpr between platforms.
        """
        np.save(f"{path_to_folder}/epsilon.npy", np.asarray(self.epsilon))
        np.save(f"{path_to_folder}/sample_points.npy", np.asarray(self.points))
        np.save(f"{path_to_folder}/alphas.npy", np.asarray(self.alphas))


def make_gaussian_process(
    X: jnp.ndarray, Y: jnp.ndarray, /, epsilon=None, sigma=None, distance=EuclideanDistance
) -> GaussianProcess:
    """Return Gaussian process regressor for given data and labels."""
    X, Y = jnp.atleast_2d(X), jnp.atleast_2d(Y)
    assert X.shape[0] == Y.shape[0]
    f = GaussianProcess(epsilon=epsilon, sigma=sigma, distance=distance)
    f.learn(X, Y)
    return f
