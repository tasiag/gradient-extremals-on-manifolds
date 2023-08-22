'''
Gradient extremal on sphere with "simple potential" (phi=xyz). 
'''
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

from SimplePotential import SimplePotential
from utils import plot_spherical_potential, plot_points3d

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.linalg import eigvalsh

from gradient_extremals_on_manifolds.Continuation import Continuation

plt.close()

#Direct input 
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
#Options
params = {'text.usetex' : True,
          'font.size' : 18
          }
plt.rcParams.update(params) 

simple_potential = SimplePotential()

fixed_points = simple_potential.get_fixed_points(manifold=True)
fixed_points_2D = jax.vmap(simple_potential.phi)(fixed_points)

initial = fixed_points_2D[0]+jnp.array([0,0.005])
hes = simple_potential.hess(initial)
lam = eigvalsh(hes)[0]

gradient_extremal = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                                          lam, simple_potential.potential(initial)]),
                                 functions = [simple_potential.lucia_phi,
                                             simple_potential.lucia_hessian_eq1,
                                             simple_potential.lucia_hessian_eq2],
                                 maxiter = 580,
                                 verbose = 0,
                                 tolerance = 0.5,
                                 h = 1E-1)


gradient_extremal.start()
gradient_extremal_points = gradient_extremal.getPoints()

initial = fixed_points_2D[0]+jnp.array([-0.05, -0.01])#[0.1,0.1])
hes = simple_potential.hess(initial)
lam = eigvalsh(hes)[0]

gradient_extremal_right = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                                                lam, simple_potential.potential(initial)]),
                                functions = [simple_potential.lucia_phi,
                                             simple_potential.lucia_hessian_eq1,
                                             simple_potential.lucia_hessian_eq2],
                                maxiter = 550,
                                verbose = 0,
                                tolerance = 0.5,
                                h = 1E-1)

gradient_extremal_right.start()
gradient_extremal_rightpoints = gradient_extremal_right.getPoints()

simple_potential.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$')
plt.plot(list(zip(*gradient_extremal_points))[0],
         list(zip(*gradient_extremal_points))[1], color="orange")

plt.plot(list(zip(*gradient_extremal_rightpoints))[0],
         list(zip(*gradient_extremal_rightpoints))[1], color="orange")

## 2D plot
plt.scatter(x=fixed_points_2D[[0,2,4],0], y=fixed_points_2D[[0,2,4],1],
            color="Yellow", zorder=15)
plt.scatter(x=fixed_points_2D[[1,3],0], y=fixed_points_2D[[1,3],1],
            color="Yellow", marker="s", s=40, zorder=20)
plt.xlabel(r"$u$")
plt.ylabel(r"$v$")

plt.show()

gradient_extremal_3D = jax.vmap(simple_potential.psi)(jnp.array(gradient_extremal_points)[:,0:2])
gradient_extremal_right_3D = jax.vmap(simple_potential.psi)(jnp.array(gradient_extremal_rightpoints)[:,0:2])

## 3D plot
plot_spherical_potential(simple_potential.E)
plot_points3d(gradient_extremal_3D, s=[ 1 for i in range(len(gradient_extremal_points))],
              color=(1.0, 0.65, 0.0))
plot_points3d(gradient_extremal_right_3D, s=[ 1 for i in range(len(gradient_extremal_rightpoints))],
              color=(1.0, 0.65, 0.0))
plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
              color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.05)
plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
              color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)

mlab.show()
