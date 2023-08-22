from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import logging

from mayavi import mlab

import sys

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

def setup_log(name, verbose):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler = logging.FileHandler(f"./{name}.log")
    if verbose:
        log_handler.setLevel(logging.DEBUG)
    else:
        log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(log_format)

    logger.addHandler(log_handler)
    logger.addHandler(stream_handler)
    return logger 

def get_direction(points, logger=None):
    ''' Returns normalized direction of line of best fit through points'''
    datamean = points.mean(axis=0)
    uu, dd, vv = jnp.linalg.svd(points - datamean)

    check_dir_vector = (points[-1]-points[0])/jnp.linalg.norm((points[-1]-points[0]))
    signs_from_utils = jnp.dot(check_dir_vector, vv[0])

    if abs(signs_from_utils) < 0.8: # bad fit, too much curvature
        if logger is not None: logger.info("Get Direction | Bad fit, using last two points")
        return check_dir_vector # instead, just use last 2 points 

    return vv[0]*jnp.sign(jnp.dot(check_dir_vector, vv[0]))

def find_first_nan(points):
    ''' Returns index of first nan in list or array. Otherwise returns length.'''
    for i in range(len(points)):
        if jnp.isnan(points[i]):
            return i
    return len(points)


