'''
Produce sphere with MullerBrown potential and a gradient extremal 
from rightmost minimum.
'''

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

from SphericalMB import SphericalMB
from utils import plot_spherical_potential, plot_points3d

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.linalg import eigvalsh

from gradient_extremals_on_manifolds.Continuation import Continuation

plt.close()

DEFAULT_VERBOSE = 1

mb = SphericalMB()
fixed_points = mb.get_fixed_points(manifold=True)
fixed_points_2D = jax.vmap(mb.phi)(fixed_points)

initial = fixed_points_2D[0]+jnp.array([0,0.03])
hes = mb.hess(initial)
lam = eigvalsh(hes)[0]

gradient_extremal = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                 lam, mb.potential(initial)]),
                                 functions = [mb.lucia_phi,
                                             mb.lucia_hessian_eq1,
                                             mb.lucia_hessian_eq2],
                                 maxiter = 2800,
                                 verbose = DEFAULT_VERBOSE,
                                 tolerance = 5,
                                 h = 2.5)


gradient_extremal.start()
gradient_extremal_points = gradient_extremal.getPoints()

mb.plot_color_mesh(colorbarTitle=r'$\psi*E$', vrangeOn=True)
plt.plot(list(zip(*gradient_extremal_points))[0],
         list(zip(*gradient_extremal_points))[1], color="orange")
plt.scatter(x=fixed_points_2D[[0,2,4],0], y=fixed_points_2D[[0,2,4],1],
            color="Yellow", marker="s", zorder=15)
plt.scatter(x=fixed_points_2D[[1,3],0], y=fixed_points_2D[[1,3],1],
            color="Yellow", marker="^", s=60, zorder=20)
plt.xlabel(r"$u$")
plt.ylabel(r"$v$")

plt.show()

gradient_extremal_3D = jax.vmap(mb.psi)(jnp.array(gradient_extremal_points)[:,0:2])

## plotting
plot_spherical_potential(mb.E)
plot_points3d(gradient_extremal_3D, s=[ 1 for i in range(len(gradient_extremal_points))],
              color=(1.0, 0.65, 0.0))
plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
              color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.05)
plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
              color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)

mlab.show()
