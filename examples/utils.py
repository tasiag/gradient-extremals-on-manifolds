from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from mayavi import mlab

DEFAULT_MESH_SIZE: int = 100
DEFAULT_AZIMUTH: int = 60
DEFAULT_ELEVATION: int = 105

DEFAULT_COLOR: Tuple[float] = (0.0, 0.0, 0.0)

def plot_spherical_potential(potential: Callable[[jnp.ndarray], float],
                             mesh_size: int = DEFAULT_MESH_SIZE,
                             azimuth: int = DEFAULT_AZIMUTH,
                             elevation: int = DEFAULT_ELEVATION) -> None:
    mlab.figure(size=(1024, 1024))
    n = mesh_size
    u1, u2 = jnp.meshgrid(jnp.linspace(0.0, 2.0 * jnp.pi, n),
                          jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n))
    x1 = jnp.cos(u1) * jnp.cos(u2)
    x2 = jnp.sin(u1) * jnp.cos(u2)
    x3 = jnp.sin(u2)
    X = jnp.stack((x1.ravel(), x2.ravel(), x3.ravel())).T

    energy = jnp.clip(jax.vmap(potential)(X), -150.0, 20.0)
    mlab.mesh(x1, x2, x3, scalars=-energy.reshape(n, n), colormap='Blues')
    mlab.view(azimuth=azimuth, elevation=elevation)

def plot_points3d(points: jnp.ndarray,
                  s=None,
                  color=DEFAULT_COLOR,
                  mode='sphere',
                  scale_factor=0.01) -> None:
    if s is not None:
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                      s, resolution=16, color=color,#map='RdBu',
                      scale_mode='none', scale_factor=scale_factor, mode=mode)
    else:
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                      color=color, resolution=16, scale_factor=scale_factor)

def plot_lines3d(points: jnp.ndarray,
                  s=None,
                  color=DEFAULT_COLOR,
                  representation = 'surface',
                  line_width=0.005) -> None:
    if s is not None:
        mlab.plot3d(points[:, 0], points[:, 1], points[:, 2],
                    s, representation=representation, color=color,#map='RdBu',
                    line_width=line_width)
    else:
        mlab.plot3d(points[:, 0], points[:, 1], points[:, 2],
                    representation=representation, color=color,#map='RdBu',
                    line_width=line_width, tube_radius = 0.005)
