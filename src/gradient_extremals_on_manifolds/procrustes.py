"""RMSD distance module."""

import numpy as np

import jax
import jax.numpy as jnp

import tqdm


@jax.jit
def procrustes(A: jnp.ndarray, B: jnp.ndarray, centered: bool=True) \
        -> jnp.ndarray:
    """Compute orthogonal matrix Q that minimizes ‖ A - B Q ‖_F.

    The points in A and B are assumed to have zero mean.

    Example
    -------
    Calling `Q = procrustes(A, B)` yields an orthogonal matrix Q such
    that `B @ Q` is the best approximation to A in the Frobenius norm.

    References
    ----------
    G. H. Golub and C. F. Van Loan, Matrix computations, Johns Hopkins
    University Press, 2013.

    """
    U, Σ, VT = jnp.linalg.svd(B.T @ A, full_matrices=False)
    return U @ VT


@jax.jit
def msd(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Compute mean square deviation of A and B."""
    #if A.ndim == 1: A = A.reshape(-1,1)
    #if B.ndim == 1: B = B.reshape(-1,1)
    Ap, Bp = A - jnp.mean(A, axis=0), B - jnp.mean(B, axis=0)
    Q = procrustes(Ap, Bp)

    return jnp.sum(jnp.linalg.norm(Ap - Bp @ Q, axis=1)**2) / Ap.shape[0]


@jax.jit
def rmsd(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Compute root mean square deviation of A and B."""
    return jnp.sqrt(msd(A, B))


def compute_msd_distance_matrix(points: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix of pairwise mean square deviations."""
    n = points.shape[0]
    print(points.shape)

    squared_distance_matrix = np.zeros((n, n))

    for i in tqdm.tqdm(range(n), desc='Computing RMSD distance matrix'):
        squared_distance_matrix[i, :] = jax.vmap(
            lambda x: msd(x, points[i, ...]))(points)

    return squared_distance_matrix

# A = jnp.array([0.4, 0.2])#, [0.3, 0.2], [0.3, 0.1]])
# B = jnp.array([0.1, 0.4])#, [0.3, 0.2],[0.23,0.2]])

# A = jnp.array([[0.2, 0.2],
#                [0.7, 0.2],
#                [0.2, 0.2],
#                [0.2, 0.3]])
# B = jnp.array([[0.3, 0.2],
#                [0.1, 0.4],
#                [0.5, 0.3],
#                [0.5, 0.3]])
# print(msd(A,B))
# print(compute_msd_distance_matrix(A))
