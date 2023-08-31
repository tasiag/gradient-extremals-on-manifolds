from abc import ABC, abstractmethod
from functools import partial
from jax import jit

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy


class Potential(ABC):
    '''Defines a potential in Euclidean space.
       DiffGeom_Potential extends this to potentials on manifolds.
    '''

    @abstractmethod
    def potential(self, z):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getTitle():
        raise NotImplementedError

    @abstractmethod
    def get_suggested_domain():
        raise NotImplementedError

    @abstractmethod
    def get_suggested_vrange(mode="potential"):
        raise NotImplementedError

    # return force of potential (jacobian)
    @partial(jit, static_argnums=0)
    def F(self, z):
        return jax.jacobian(self.potential)(z)

    # returns squared magnitude of force (derivative)
    @partial(jit, static_argnums=0)
    def FTF(self, z):
        return jnp.sqrt(jnp.dot(self.F(z), self.F(z)))

    @partial(jit, static_argnums=0)
    def hessian(self, z):
        return jax.jacfwd(jax.jacrev(self.potential))(z)

    # z = x, y, ..., lambda, L
    @partial(jit, static_argnums=0)
    def lucia_phi(self, z):
        return self.potential(z[0:-2])-z[3]

    # z = x, y, ..., lambda, L
    @partial(jit, static_argnums=0)
    def lucia_hessian_eq1(self, z):
        hess = self.hessian(z[0:-2])
        grad = self.F(z[0:-2])
        return jnp.dot(hess[0], grad) - z[2]*grad[0]

    # z = x, y, ... lambda, L
    @partial(jit, static_argnums=0)
    def lucia_hessian_eq2(self, z):
        hess = self.hessian(z[0:-2])
        grad = self.F(z[0:-2])
        #jax.debug.print("eq_2: {x}", x=jnp.dot(hess[1], grad) - z[2]*grad[1])
        return jnp.dot(hess[1], grad) - z[2]*grad[1]

    def plot_gradient_field(self,
                            no_points=(30, 30),
                            fig=plt.figure(),
                            domain=None):
        if domain == None: domain = self.get_suggested_domain()
        xxq = jnp.linspace(domain[0], domain[1], no_points[0])
        yyq = jnp.linspace(domain[2], domain[3], no_points[1])
        xq, yq = jnp.meshgrid(xxq, yyq)
        Xq = jnp.stack((xq.ravel(),yq.ravel())).T

        F=jax.vmap(self.F)(Xq)
        F=F/jnp.linalg.norm(F,axis=1)[...,None]
        plt.quiver(Xq[:,0], Xq[:,1],-F[:,0],-F[:,1])
        return fig

    def plot_color_mesh(self,
                        no_points=(600,600),
                        fig=plt.figure(),
                        domain = None,
                        contour = False,
                        colorMesh = True,
                        contourCount = 80,
                        colorbarTitle = None,
                        contourcolors = "black",
                        vrangeOn = False):
        if domain is None: domain = self.get_suggested_domain()
        vrange = self.get_suggested_vrange()
        xx = jnp.linspace(domain[0], domain[1], 600)
        yy = jnp.linspace(domain[2], domain[3], 600)

        x, y = jnp.meshgrid(xx, yy)
        X = jnp.stack((x.ravel(),y.ravel())).T

        FTF_value = jax.vmap(self.FTF)(jnp.array(X)).reshape(jnp.shape(x))
        pot_value = jax.vmap(self.potential)(jnp.array(X)).reshape(jnp.shape(x))

        if colorMesh:
            if vrangeOn:
                plt.pcolormesh(x, y, pot_value, cmap="Blues_r", shading="auto", vmax=vrange[1])
            else:
                plt.pcolormesh(x, y, pot_value, cmap="Blues_r", shading="auto")

            c = plt.colorbar()
            if colorbarTitle is not None:
                c.set_label(colorbarTitle)

        if contour and contourcolors != "black":
            plt.contour(x, y, pot_value, contourCount, colors=contourcolors)
        elif contour:
            plt.contour(x, y, pot_value, contourCount, colors="black", linestyles='solid')
        return fig

    def plot_surface(self,
                     no_points=(600,600),
                     domain = None,
                     Line=None):

        if domain is None: domain = self.get_suggested_domain()
        vrange = self.get_suggested_vrange("potential2")
        xx = jnp.linspace(domain[0], domain[1], no_points[0])
        yy = jnp.linspace(domain[2], domain[3], no_points[1])

        x, y = jnp.meshgrid(xx, yy)
        X = jnp.stack((x.ravel(),y.ravel())).T

        ax = plt.axes(projection ='3d')
        pot_value = jax.vmap(self.potential)(jnp.array(X)).reshape(jnp.shape(x))
        ax.plot_surface(x, y, pot_value,
                        vmin=vrange[0],
                        vmax=vrange[1],
                        cmap='Blues',
                        alpha=0.5)
        ax.contour3D(x, y, pot_value, 20, alpha=0.6)

        if Line is not None:
            for _, line in Line.items():
                print("getting line")
                ax.plot3D(line[0][0], line[0][1], line[0][2]+abs(line[0][2])/50,
                          color=line[1],
                          alpha=1,
                          ms=20,
                          label=line[2],
                          linewidth=2)

        ax.legend()

        ax.set_xlim(domain[0], domain[1])
        ax.set_ylim(domain[2], domain[3])
        ax.set_zlim(vrange[0], vrange[1])

    def find_equilibrium(self, initial_guess):
        """Find equilibrium point near a given initial guess."""
        res = scipy.optimize.minimize(self.potential,
                                      x0=initial_guess, method='Newton-CG',
                                      jac=jax.jacobian(self.potential),
                                      hess=jax.hessian(self.potential))
        return res.x
