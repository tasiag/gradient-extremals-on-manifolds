from functools import partial
from jax import jit
import jax.numpy as jnp

from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential

class SphericalMB(DiffGeom_Potential):
    '''
    Potential on a sphere with four local minima, with equations copied from
    the paper "Gentlest Ascent Dynamics on Manifolds defined by Sampled Point Clouds"
    '''

    def __init__(self):
        super().__init__(self.pullback_energy,
                         self.phi,
                         self.psi)

    # z = [x1, x2, x3]
    @partial(jit, static_argnums=0)
    def potential2d(self, z):
        """Original Muller potential."""
        return -200.0*jnp.exp(-(z[0]-1.0)**2-10.0*z[1]**2)\
                -100.0*jnp.exp(-z[0]**2-10.0*(z[1]-1/2)**2)\
                -170.0*jnp.exp(-(13/2)*(z[0]+1/2)**2+11.0*(z[0]+1/2)*(z[1]-3/2)-(13/2)*(z[1]-3/2)**2)\
                +15.0*jnp.exp((7/10)*(z[0]+1)**2+(3/5)*(z[0]+1.0)*(z[1]-1.0)+(7/10)*(z[1]-1.0)**2)

    # u = [u, v]
    @partial(jit, static_argnums=0)
    def polar_to_cartesian(self, u):
        return jnp.array([1.973521294 * u[0] - 1.85, 1.750704373 * u[1] + 0.875])

    # z = [x1, x2, x3]
    @partial(jit, static_argnums=0)
    def E(self, z):
        u = jnp.array([jnp.arctan2(z[1], z[0]),
                   jnp.arctan2(z[2], jnp.sqrt(z[0]**2 + z[1]**2))])

        return self.potential2d(self.polar_to_cartesian(u))

    # u = [u, v]
    @partial(jit, static_argnums=0)
    def pullback_energy(self, u):
        return self.E(self.psi(u))

    # pullback from 2D chart (inverse stereographic potential)
    @partial(jit, static_argnums=0)
    def psi(self, u):
        x = 2.0*u[0] / (1.0 + u[0]**2 + u[1]**2)
        y = 2.0*u[1] / (1.0 + u[0]**2 + u[1]**2)
        z = (-1.0 + u[0]**2 + u[1]**2) / (1.0 + u[0]**2 + u[1]**2)
        return jnp.array([x, y, z])

    # pushforward from 3D to 2D chart (stereographic projection from North pole onto
    # tangent plane of South pole)
    @partial(jit, static_argnums=0)
    def phi(self, z):
        return jnp.array([z[0] / (1.0-z[2]), z[1] / (1.0-z[2])])

    @staticmethod
    def getTitle():
        return "Spherical MB Potential"

    # [xmin, xmax, ymin, ymax]
    def get_suggested_domain(self):
        return jnp.array([0.0, 2.0, 0.0, 2.0])

    @staticmethod
    def get_suggested_vrange(mode="potential"):
        return jnp.array([-150.0, 25.0])

    # Fixed points of the potential in Chart 2D coordinates.
    def get_fixed_points(self, manifold=False):
        muller_fixed_points3d = jnp.array(
            [[0.27632632, 0.84100892, -0.46513199],
            [0.47435655, 0.81760542, -0.32635446],
            [0.59553828, 0.76936331, -0.23111522],
            [0.85849672, 0.49256384, -0.14270336],
            [0.75211518, 0.5771847, 0.31808894]])

        if manifold:
            return muller_fixed_points3d

        return jnp.array([self.phi(x) for x in muller_fixed_points3d])
