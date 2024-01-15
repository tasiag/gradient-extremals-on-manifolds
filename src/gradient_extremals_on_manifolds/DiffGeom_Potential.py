from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable, Tuple

from .Potential import Potential


class DiffGeom_Potential(Potential):
    '''
    Learnt local potential on adaptively sampled point cloud. 
    '''

    def __init__(self,
                 potential: Callable,   # pullback energy, psi*E
                 phi: Callable,         # mapping from ambient -> latent
                 psi: Callable):        # mapping from latent -> ambient):

        self.potential_func = potential # this should take in a point in latent space
        self.phi_func = phi # ambient -> latent
        self.psi_func = psi # latent -> ambient

    @partial(jit, static_argnums=0)
    def potential(self, u: jnp.ndarray):
        '''
        Evaluate pullback energy at position u (position on local chart in reduced 
        coordinates). Returns scalar.
        '''
        return self.potential_func(u).squeeze()

    @partial(jit, static_argnums=0)
    def hess(self, u: jnp.ndarray):
        '''
        Evaluate Hessian defined in terms of covariant derivative at position u
         (position on local chart in reduced coordinates). Returns uxu array.
        '''
        shape = len(u)
        ch = self.christoffel(u)
        g = self.grad(u)
        h = jax.jacobian(self.grad)(u)
        for k in range(shape):
            for i in range(shape):
                correction = ch[k,i,0]*g[0] + ch[k,i,1]*g[1]
                h = h.at[k,i].set(h[k,i]+correction)
        return h

    @partial(jit, static_argnums=0)
    def metric(self, u: jnp.ndarray) -> jnp.array:
        '''
        Evaluate Riemannian metric at position u (position on local chart 
        in reduced coordinates). Returns uxu tensor.
        '''
        return jax.jacobian(self.psi)(u).T @ jax.jacobian(self.psi)(u)

    @partial(jit, static_argnums=0)
    def inverse_metric(self, u: jnp.ndarray) -> jnp.array:
        '''
        Inverse metric tensor at position u (position on local chart 
        in reduced coordinates). Returns uxu tensor.
        '''
        return jnp.linalg.inv(self.metric(u))

    @partial(jit, static_argnums=0)
    def christoffel(self, u: jnp.array) -> jnp.array:
        '''
        Inverse metric tensor at position u (position on local chart 
        in reduced coordinates). Returns uxu tensor.
        '''
        shape = len(u)
        ch = jnp.zeros((shape,shape,shape))
        jacobian = jax.jacobian(self.metric)(u)

        inverse = self.inverse_metric(u)

        for k in range(shape):
            for i in range(shape):
                for j in range(shape):
                    summation = 0
                    for l in range(shape):
                        summation += inverse[k,l]* \
                                     (jacobian[j,l,i] + jacobian[i,l,j] - jacobian[i,j,l])
                    ch = ch.at[k,i,j].set(0.5*summation)
        return ch


    @partial(jit, static_argnums=0)
    def grad(self, u: jnp.array) -> jnp.array:
        return self.inverse_metric(u) @ jax.jacobian(self.potential)(u)

    # pullback from 2D chart (inverse stereographic potential)
    @partial(jit, static_argnums=0)
    def psi(self, u):
        return self.psi_func(u)

    # pushforward from 3D to 2D chart (stereographic projection from North pole onto
    # tangent plane of South pole)
    @partial(jit, static_argnums=0)
    def phi(self, z: jnp.array) -> jnp.array:
        return self.phi_func(z)

    # x-nullcline
    @partial(jit, static_argnums=0)
    def x_nullcline(self, u: jnp.array, value = 0):
        return self.grad(u)[0] - value

    # y-nullcline
    @partial(jit, static_argnums=0)
    def y_nullcline(self, u: jnp.array, value = 0):
        return self.grad(u)[1] - value

    # equal-nullcline
    @partial(jit, static_argnums=0)
    def equal_nullcline(self, u: jnp.array):
        return self.grad(u)[1] - self.grad(u)[0]

    # p = u, v, lambda, L
    @partial(jit, static_argnums=0)
    def lucia_phi(self, p: jnp.array):
        answer = self.potential(p[0:-2])-p[-1]
        return answer

    # p = u, v, lambda, L
    @partial(jit, static_argnums=0)
    def lucia_hessian_eq1(self, p: jnp.array):
        h = self.hess(p[0:-2])
        g = self.grad(p[0:-2])
        return jnp.dot(h[0], g) - p[-2]*g[0]

    # p = u, v, lambda, L
    @partial(jit, static_argnums=0)
    def lucia_hessian_eq2(self, p: jnp.array):
        h = self.hess(p[0:-2])
        g = self.grad(p[0:-2])
        return jnp.dot(h[1], g) - p[-2]*g[1]

    # this is for plotting purposes
    @staticmethod
    def getTitle():
        return ""

    #xmin, xmax, y1, y2
    # this is for plotting purposes for given examples.
    def get_suggested_domain(self):
        # make this smart, where is phi(x) bad?
        return jnp.array([ -0.002, 0.002, -0.002, 0.002])

    # this is for plotting purposes for given examples.
    @staticmethod
    def get_suggested_vrange(mode="potential"):
        return jnp.array([-150.0, 25.0])
