from typing import Optional

import jax.numpy as jnp

def compute_median(matrix: jnp.ndarray) -> float:
    """Return median of the upper triangle of a symmetric matrix.

    """
    n = matrix.shape[0]
    i, j = zip(*[(i, j) for i in range(n) for j in range(i + 1, n)])
    return jnp.median(matrix[i, j])


def guess_epsilon(squared_distance_matrix: jnp.ndarray,
                  epsilon: Optional[float] = None,
                  threshold: Optional[float] = None) -> float:
    """Guess spatial scale for diffusion maps and Gaussian processes.

    Computes the pairwise distance between points.

    """
    if epsilon is not None:
        return epsilon
    elif threshold is not None:
        distances2 = jnp.tril(squared_distance_matrix)
        print("LENGTH HERE: ", len(distances2[distances2 > threshold]))
        for i in range(10):
            if len(distances2[distances2 > threshold]) < 1:
                print("dividing it in half: ")
                threshold = threshold / 2
                print("new length: ", len(distances2[distances2 > threshold]))
            else: 
                break
        return jnp.sqrt(jnp.median(distances2[distances2 > threshold]))
    else:
        return jnp.sqrt(compute_median(squared_distance_matrix))