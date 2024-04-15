'''
Code written by Juan Bello-Rivas

https://github.com/jmbr/staying-the-course-code/blob/main/
        src/staying_the_course/diffusion_map_coordinates.py
'''

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from .diffusion_maps import DiffusionMaps
from .gaussian_process import GaussianProcess
from .distances import RMSDDistance, EuclideanDistance, Distance


class DiffusionMapCoordinates():
    def __init__(self, domain_dimension: int, codomain_dimension: int, distance: Distance = EuclideanDistance) -> None:
        self.domain_dimension = domain_dimension
        self.codomain_dimension = codomain_dimension
        self.diffusion_maps = DiffusionMaps(codomain_dimension, distance=distance)
        self.gaussian_process = GaussianProcess(distance=distance)

    def learn(self, points: jnp.ndarray) -> None:
        self.diffusion_maps.learn(points)
        coordinates = (self.diffusion_maps._eigenvalues
                       * self.diffusion_maps._eigenvectors)
        self.gaussian_process._learn(points, coordinates,
                                     self.diffusion_maps.epsilon,
                                     self.diffusion_maps._kernel_matrix)
        return coordinates

    @partial(jax.jit, static_argnums=0)
    def __call__(self, point: jnp.ndarray) -> jnp.ndarray:
        return self.gaussian_process(point)

    def save_gpr(self, path_to_folder):
        print(f"!!!!!!!!! saving gpr, and the alphas have shape: {self.gaussian_process.alphas.shape}")

        try:
            with open(f"{path_to_folder}/epsilon.csv", "w") as file:
                print("EPSILON: ", self.gaussian_process.epsilon)
                file.write(f"{self.gaussian_process.epsilon}\n")
            np.savetxt(f"{path_to_folder}/alphas.csv", np.asarray(self.gaussian_process.alphas), delimiter=",")
            return True
        except Exception as e:
            print(e)
            return False

