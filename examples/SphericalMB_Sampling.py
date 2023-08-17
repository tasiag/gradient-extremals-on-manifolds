'''
Full implementation of algorithm (sampling, manifold learning, differential geometry)
for tracing gradient extremals on the Muller Brown Potential mapped onto a sphere
by using sampled point clouds.
'''
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

import jax
import jax.numpy as jnp

import logging

from SphericalMB import SphericalMB
from utils import plot_spherical_potential, plot_points3d, plot_lines3d, setup_log

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from mayavi import mlab
import numpy as np

from scipy.linalg import eigvalsh
from scipy.spatial import ConvexHull

from gradient_extremals_on_manifolds.Continuation import Continuation
from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential
from gradient_extremals_on_manifolds.DiffusionMapCoordinates import DiffusionMapCoordinates
from gradient_extremals_on_manifolds.gaussian_process import make_gaussian_process
from gradient_extremals_on_manifolds.Sampler import Sampler


DEFAULT_VERBOSE = 0 # 0 for INFO only, 1 for DEBUG
DEFAULT_PLOT = 1
DEFAULT_SAVE = 1

DEFAULT_ITERATIONS = 18
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_TOLERANCE = 1

plt.close()

#Direct input 
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
#Options
params = {'text.usetex' : True,
          'font.family' : 'lmodern',
          'font.size' : 18
          }
plt.rcParams.update(params) 


mb = SphericalMB()
fixed_points = mb.get_fixed_points(manifold=True)
fixed_points_2D = jax.vmap(mb.phi)(fixed_points)

path_3D = None
initial_3D = fixed_points[0]

logger = setup_log("sphericalMB_sampling", DEFAULT_VERBOSE)

for i in range(DEFAULT_ITERATIONS):

    sampler = Sampler(mb.E, lambda x: x[0]**2+x[1]**2+x[2]**2 - 1, noise_level=0.05)
    final_3D = sampler.draw_samples(initial_3D, DEFAULT_SAMPLE_SIZE)

    if DEFAULT_PLOT:
        plot_spherical_potential(mb.E)
        plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
                      color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.05)
        plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
                      color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)
        plot_points3d(final_3D, s=[ 1 for i in range(final_3D.shape[0])], color=(1.0, 0.65, 0.0))
        plt.tight_layout()
        if DEFAULT_SAVE:
            mlab.savefig(str(i)+"_Samples.png")
        mlab.show()
        mlab.close(all=True)

    # obtain energies at samples
    energies = np.expand_dims(jax.vmap(mb.E)(final_3D), axis=1)
    ambient_dim = final_3D.shape[1]

    # learn pushforward
    phi = DiffusionMapCoordinates(ambient_dim, 2) # dimension of manifold set
    final_2D = phi.learn(final_3D)

    logger.debug("domain: " + str(np.min(final_2D[:,0])) + " to " + \
                 str(np.max(final_2D[:,0])) + ", " + str(np.min(final_2D[:,1])) \
                 + " to " + str(np.max(final_2D[:,1])))
    logger.debug("Learned_phi: " + str(phi(final_3D[0])))
    logger.debug("true phi: " + str(final_2D[0]))

    # learn pullback
    psi = make_gaussian_process(jnp.asarray(final_2D), jnp.asarray(final_3D))
    logger.debug("Learned_psi: " + str(psi(final_2D[0])))
    logger.debug("true psi: " + str(final_3D[0]))

    # get convex hull in 2D & 3D just for plotting purposes
    hull = ConvexHull(final_2D)
    boundary = jnp.concatenate((hull.vertices, jnp.array([hull.vertices[0]])))

    # create the pullback potential
    learned_potential_func = make_gaussian_process(final_2D, energies)
    learned_potential = DiffGeom_Potential(learned_potential_func, phi, psi)
    logger.debug("LEARNED_ POTENTIAL GP: " + str(learned_potential_func(final_2D[0])))
    logger.debug("TRUTH: " + str(energies[0]))
    logger.debug("LEARNED_POTENTIAL: " + str(learned_potential.potential_func(final_2D[0])))

    # move some small distance away from minimum to choose direction
    initial = phi(initial_3D) + jnp.array([0.0000,0.00003])
    hes = learned_potential.hess(initial)
    lam = eigvalsh(hes)[0]

    logger.debug("initial: " + str(initial))
    logger.debug("hes: " + str(hes))
    logger.debug("lam: " + str(lam))
    logger.debug("variance about point: " + \
                 str(learned_potential_func.get_variance(initial)))

    # run gradient extremals
    gradient_extremal = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                                              lam,
                                                              learned_potential.potential(initial)]),
                                     functions = [learned_potential.lucia_phi,
                                                  learned_potential.lucia_hessian_eq1,
                                                  learned_potential.lucia_hessian_eq2],
                                     maxiter = 150,
                                     verbose = 1,
                                     tolerance = 1,
                                     h = 5)
    # this stops at 150 iterations; could also stop when variance is above threshold
    # using the following line
    # max_cond = lambda x:learned_potential.potential_func.get_variance(x[0:2]) > 2E-6 (alternate)

    gradient_extremal.start()
    gradient_extremal_points = gradient_extremal.getPoints()

    if DEFAULT_PLOT:
        learned_potential.plot_color_mesh(colorbarTitle=r'$Z:=\psi*E$', contour=True, contourCount=20)
        
        # plot minimums and saddles
        fixed_points_on_chart = jax.vmap(phi)(fixed_points)

        # need to get in bound indices just for plotting nicely
        if i==1: 
            plt.scatter(fixed_points_on_chart[[0],0],
                        fixed_points_on_chart[[0],1],
                        color = (1.0, 1.0, 0.0))

        # plt convex hull in 2D
        plt.plot(final_2D[boundary,0], final_2D[boundary,1], 'r--', lw=2)

        plt.plot(list(zip(*gradient_extremal_points))[0], list(zip(*gradient_extremal_points))[1],
                 color="orange", linewidth=0.5)
        plt.scatter(list(zip(*gradient_extremal_points))[0][::15],
                    list(zip(*gradient_extremal_points))[1][::15],
                    color="orange", s=12, zorder=30)
        ax = plt.gca()
        loc = plticker.MultipleLocator(base=0.001) # this locator puts ticks at regular intervals
        ax.axes.xaxis.set_major_locator(loc)
        ax.axes.yaxis.set_major_locator(loc)
        plt.xlabel(r"$u$")
        plt.ylabel(r"$v$")
        plt.tight_layout()
        if DEFAULT_SAVE:
            plt.savefig(str(i)+"_2D.png")
        plt.show()

    gradient_extremal_3D = jax.vmap(learned_potential.psi)(jnp.array(gradient_extremal_points)[:,0:2])
    if path_3D is None:
        path_3D = gradient_extremal_3D
    else:
        path_3D = jnp.concatenate((path_3D, gradient_extremal_3D), axis=0)

    if DEFAULT_PLOT:
        plot_spherical_potential(mb.E)
        plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
                      color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)
        plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
                      color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.03)
        plot_points3d(gradient_extremal_3D, s=[ 1 for i in range(len(gradient_extremal_points))],
                      color=(1.0, 0.65, 0.0), scale_factor=0.02)
        plot_lines3d(final_3D[boundary,:], color = (1.0, 0.0, 0.0))

        plt.tight_layout()
        if DEFAULT_SAVE:
            mlab.savefig(str(i)+"_3D.png")
        mlab.show()
        mlab.close(all=True)

    initial_3D = gradient_extremal_3D[-2]

    logger.info("NEW POINT: " + str(initial_3D))
    logger.info("CRITERIA TO END EARLY: " + str(jnp.linalg.norm(jax.jacobian(mb.E)(initial_3D))))

    if jnp.linalg.norm(jax.jacobian(mb.E)(initial_3D)) < DEFAULT_TOLERANCE:
        break

## plotting final solution
plot_spherical_potential(mb.E)
plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
              color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)
plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
              color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.03)
plot_points3d(path_3D, s=[ 1 for i in range(len(path_3D))], color=(1.0, 0.65, 0.0))
plt.tight_layout()
if DEFAULT_SAVE:
    mlab.savefig("Final_3D.png")
mlab.show()
