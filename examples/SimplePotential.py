from functools import partial
from jax import jit
import jax.numpy as jnp

from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential

class SimplePotential(DiffGeom_Potential):
    '''
    Potential on a sphere with four local minima, copied from
    the paper "Gentlest Ascent Dynamics on Manifolds defined by Sampled Point Clouds"
    '''
    def __init__(self):
        super().__init__(self.potential,
                         self.phi,
                         self.psi)

    # z = [x, y, z]
    @partial(jit, static_argnums=0)
    def E(self, z):
        return z[0]*z[1]*z[2]

    # u = [u, v]
    @partial(jit, static_argnums=0)
    def potential(self, u):
        return 100*u[0]*u[1]*(u[0]**2 + u[1]**2 -1.0) /  (u[0]**2 + u[1]**2 + 1.0)**3

    # u = [u, v]
    @partial(jit, static_argnums=0)
    def grad(self, u):
        g = jnp.array([u[1]**5 - 2.0*u[0]**2*u[1]**3 - 3.0*u[0]**4*u[1] + 8.0*u[0]**2*u[1] - u[1],
                       -3.0*u[0]*u[1]**4 - 2.0*u[0]**3*u[1]**2 + 8.0*u[0]*u[1]**2 + u[0]**5 - u[0]])
        return 25.0 * g / (1.0 + u[0]**2 + u[1]**2)**2

    # u = [u, v]
    @partial(jit, static_argnums=0)
    def hess(self, u):
        h11 = -4.0*u[0]*u[1]*(u[1]**4 + u[1]**2 - u[0]**4 + 11.0*u[0]**2 - 6.0)
        h12 = -(u[1]**6 - 5.0*u[0]**2*u[1]**4 - 5.0*u[1]**4 - 5.0*u[0]**4*u[1]**2 \
              + 30.0*u[0]**2*u[1]**2 - 5.0*u[1]**2 + u[0]**6 - 5.0*u[0]**4 - 5.0*u[0]**2 + 1.0)
        h21 = h12
        h22 =  4.0*u[0]*u[1]*(u[1]**4 - 11.0*u[1]**2 - u[0]**4 - u[0]**2 + 6.0)

        return 25.0*jnp.array([[h11, h12], [h21,  h22]]) / (1.0 + u[0]**2 + u[1]**2)**3

    @partial(jit, static_argnums=0)
    def metric(self, u):
        g = jnp.zeros((2, 2))
        g[0,0] = 100.0 / (1.0 + u[0]**2 + u[1]**2)**2
        g[1,1] = g[0,0]

        return g

    # pullback from 2D chart (inverse stereographic potential from North pole onto
    # tangent plane of South ole)
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


    # 2D explicit formulation
    @partial(jit, static_argnums=0)
    def gradient_extremal(self, z):
        h = self.hess(z)
        g = self.F(z)
        return h[0,1]*(g[0]**2 - g[1]**2) + (h[1,1] - h[0,0])*g[0]*g[1]

    @staticmethod
    def getTitle():
        return "Zorro Potential"

    #xmin, xmax, y1, y2
    def get_suggested_domain(self):
        return jnp.array([-1.0, 2.0, -2.0, 1.0])

    @staticmethod
    def get_suggested_vrange(mode="potential"):
        return jnp.array([-25.0, 25.0])

    # Fixed points of the potential in Chart 2D coordinates.
    def get_fixed_points(self, manifold=False):

        min_1 = jnp.array([0.57735924, -0.57734121, 0.57735039])
        sp_1  = jnp.array([1.0, 0.0, 0.0])
        min_2 =  jnp.array([0.57735451, 0.57734232, -0.577354])
        sp_2 = jnp.array([0.0, -1.0, 0.0])
        min_3 =  jnp.array([-0.57735451, -0.57734232, -0.577354])

        if manifold: return jnp.array([min_1, sp_1, min_2, sp_2, min_3])

        return jnp.array([ self.phi(min_1), # min
                           self.phi(sp_1), # saddle
                           self.phi(min_2),
                           self.phi(sp_2),
                           self.phi(min_3)]) # min
