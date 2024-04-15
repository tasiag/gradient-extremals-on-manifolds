'''
Full implementation of algorithm (sampling, manifold learning, differential geometry)
for tracing gradient extremals on the Muller Brown Potential mapped onto a sphere
by using sampled point clouds.
'''

import jax
import jax.numpy as jnp

import logging

from SphericalMB import SphericalMB
from utils import plot_spherical_potential, plot_points3d, plot_lines3d, \
                  setup_log, get_direction, find_first_nan

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import math

from mayavi import mlab

from scipy.linalg import eigvalsh
from scipy.spatial import ConvexHull

from gradient_extremals_on_manifolds.Continuation import Continuation
from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential
from gradient_extremals_on_manifolds.DiffusionMapCoordinates import DiffusionMapCoordinates
from gradient_extremals_on_manifolds.gaussian_process import make_gaussian_process
from gradient_extremals_on_manifolds.Sampler import Sampler

print("I started.")
# Options (True or False)
DEFAULT_VERBOSE = True # False for INFO only, True for DEBUG
DEFAULT_PLOT = True
DEFAULT_SHOWPLOT = True
DEFAULT_SAVE = True

# Problem-specific parameters to set
DEFAULT_ITERATIONS = 30
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_TOLERANCE = 10
DEFAULT_THRESHOLD_SHRINKSTEPSIZE = 150 # shrink step size when approach saddle
DEFAULT_MINCHARTSEXPECTED = 4
DEFAULT_SEED = 9521

plt.close()

# Text formatting
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.family' : 'lmodern',
          'font.size' : 18
          }
plt.rcParams.update(params) 

mb = SphericalMB()
fixed_points = mb.get_fixed_points(manifold=True)
fixed_points_2D = jax.vmap(mb.phi)(fixed_points)

key = None
path_3D = None
initial_3D = fixed_points[0]

print("Setting up logger.")
logger = setup_log("sphericalMB_sampling", DEFAULT_VERBOSE)

for i in range(DEFAULT_ITERATIONS):
    logger.info(f"ITERATION: {i}")

    if key == None: 
        key = jax.random.PRNGKey(DEFAULT_SEED)
    else:
        _, key = jax.random.split(key)

    print("drawing samples.")
    sampler = Sampler(mb.E, lambda x: x[0]**2+x[1]**2+x[2]**2 - 1, noise_level=0.05)
    final_3D = sampler.draw_samples(initial_3D, DEFAULT_SAMPLE_SIZE, key = key)
    #final_3D_formatted = final_3D.reshape((DEFAULT_SAMPLE_SIZE, 1, 3))
    #print("FINAL_3D_formatted.SHAPE :", final_3D_formatted.shape)
    #input()

    logger.debug(f"3d shape: {final_3D.shape}")
    logger.debug(f"Are there nan's? {jnp.any(jnp.isnan(final_3D))}")
    logger.debug(f" or None's in final_3D? ({any(None in sub for sub in final_3D)}) ")

    if DEFAULT_PLOT:
        print("plotting.")
        plot_spherical_potential(mb.E)
        plot_points3d(fixed_points[[0,2,4],:], s=[ 1 for i in range(len(fixed_points[[0,2,4],:]))],
                      color = (1.0, 1.0, 0.0), mode="sphere", scale_factor=0.05)
        plot_points3d(fixed_points[[1,3],:], s=[ 1 for i in range(len(fixed_points[[1,3],:]))],
                      color = (1.0, 1.0, 0.0), mode="cube", scale_factor=0.05)
        plot_points3d(final_3D, s=[ 1 for i in range(final_3D.shape[0])], color=(1.0, 0.65, 0.0))
        plt.tight_layout()
        if DEFAULT_SAVE:
            mlab.savefig(f"{i}_Samples.png")
        if DEFAULT_SHOWPLOT: mlab.show()
        mlab.close(all=True)

    # obtain energies at samples
    energies = jnp.expand_dims(jax.vmap(mb.E)(final_3D), axis=1)
    ambient_dim = final_3D.shape[1]

    # learn pushforward
    print("Diffusion maps.")
    phi = DiffusionMapCoordinates(ambient_dim, 2) # dimension of manifold set
    print(final_3D.shape)
    final_2D = phi.learn(final_3D) #final_3D)

    logger.debug(f"Are there nan's? {jnp.any(jnp.isnan(final_2D))}")
    logger.debug(f" or None's in final_2D? ({any(None in sub for sub in final_2D)}) ")

    logger.debug(f"domain: {jnp.min(final_2D[:,0])} to {jnp.max(final_2D[:,0])}; \
                  {jnp.min(final_2D[:,1])} to {jnp.max(final_2D[:,1])}")
    logger.debug(f"Learned_phi: {phi(final_3D[0])}")
    logger.debug(f"true phi: {final_2D[0]}")

    # learn pullback
    psi = make_gaussian_process(jnp.asarray(final_2D), jnp.asarray(final_3D))

    # get convex hull in 2D & 3D just for plotting purposes
    hull = ConvexHull(final_2D)
    boundary = jnp.concatenate((hull.vertices, jnp.array([hull.vertices[0]])))

    # create the pullback potential
    learned_potential_func = make_gaussian_process(final_2D, energies)
    learned_potential = DiffGeom_Potential(learned_potential_func, phi, psi)

    initial = phi(initial_3D)
    print(initial_3D) 
    print("initial ", initial)
    lam, vec = jnp.linalg.eigh(learned_potential.hess(initial))
    lam = lam[0]
    if i == 0: 
        initial += vec[0]*0.0003
        lam = eigvalsh(learned_potential.hess(initial))[0]

    logger.debug(f"initial: {initial}")
    logger.debug(f"lam: {lam}")
    logger.debug(f"variance about point: {learned_potential_func.get_variance(initial)}")

    h = 5

    # shrink step size when approach saddle
    if i > DEFAULT_MINCHARTSEXPECTED and jnp.linalg.norm(jax.jacobian(mb.E)(initial_3D)) < DEFAULT_THRESHOLD_SHRINKSTEPSIZE: 
        h = 2.5

    print("deciding which direction.")
    # small test continuation to decide which direction to continue 
    if i>0:
        tester = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                                       lam, learned_potential.potential(initial)]),
                                            functions = [learned_potential.lucia_phi,
                                                      learned_potential.lucia_hessian_eq1,
                                                      learned_potential.lucia_hessian_eq2],
                                            maxiter = 5,
                                            verbose = 1,
                                            tolerance = 1,
                                            h = h)
        tester.start()
        test_points = jnp.array(tester.getPoints())[:,0:2]
        test_vector = get_direction(test_points)

        prev_points_transferred = jax.vmap(phi)(jnp.array(gradient_extremal_3D)[-5:, :])
        prev_vector = get_direction(prev_points_transferred)

        sign = jnp.sign(jnp.dot(test_vector, prev_vector))
        logger.debug(f"sign checker: {jnp.dot(test_vector, prev_vector)}")
        if sign == 0 or sign == math.isnan(sign): sign = 1
        h=h*sign

        if DEFAULT_PLOT:
            learned_potential.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$', contour=True, contourCount=20)

            plt.plot(final_2D[boundary,0], final_2D[boundary,1], 'r--', lw=2)


            plt.plot(list(zip(*test_points))[0], list(zip(*test_points))[1],
                 color="orange", linewidth=0.5)
            plt.scatter(list(zip(*test_points))[0],
                        list(zip(*test_points))[1],
                        color="orange", s=12, zorder=30)
            plt.scatter(list(zip(*test_points))[0][-1],
                        list(zip(*test_points))[1][-1],
                        color="red", s=12, zorder=38)
            plt.scatter(list(zip(*test_points))[0][0],
                        list(zip(*test_points))[1][0],
                        color="purple", s=12, zorder=38)

            plt.plot(list(zip(*prev_points_transferred))[0],
                    list(zip(*prev_points_transferred))[1],
                    color="yellow", linewidth=0.5)
            plt.scatter(list(zip(*prev_points_transferred))[0],
                        list(zip(*prev_points_transferred))[1],
                        color="yellow", s=12, zorder=30)
            plt.scatter(list(zip(*prev_points_transferred))[0][-1],
                        list(zip(*prev_points_transferred))[1][-1],
                        color="red", s=12, zorder=38)
            plt.scatter(list(zip(*prev_points_transferred))[0][0],
                        list(zip(*prev_points_transferred))[1][0],
                        color="purple", s=12, zorder=38)

            ax = plt.gca()
            loc = plticker.MultipleLocator(base=0.001) # this locator puts ticks at regular intervals
            ax.axes.xaxis.set_major_locator(loc)
            ax.axes.yaxis.set_major_locator(loc)
            plt.xlabel(r"$u$")
            plt.ylabel(r"$v$")
            plt.tight_layout()
            if DEFAULT_SAVE: plt.savefig(f"{i}_Debug.png")
            if DEFAULT_SHOWPLOT: plt.show()
            plt.close()

    logger.debug(f"step size: {h}")

    # run gradient extremals
    print("running gradient extremals.")
    print(learned_potential.potential(initial))
    print("again")
    try:
        gradient_extremal = Continuation(initial_point=jnp.array([initial[0], initial[1],
                                                                  lam,
                                                                  learned_potential.potential(initial)]),
                                         functions = [learned_potential.lucia_phi,
                                                      learned_potential.lucia_hessian_eq1,
                                                      learned_potential.lucia_hessian_eq2],
                                         maxiter = 150,
                                         max_cond = lambda x:learned_potential.potential_func.get_variance(x[0:2]) > 2E-6,
                                         verbose = 1,
                                         tolerance = 1,
                                         h = h)
        # this stops at 150 iterations; could also stop when variance is above threshold
        # using the following line
        # max_cond = lambda x:learned_potential.potential_func.get_variance(x[0:2]) > 2E-6 (alternate)

        gradient_extremal.start()
    except Exception as e:
        logger.warning("SphericalMB_Sampling | Continuation failed.")
        print(Exception)

    gradient_extremal_points = gradient_extremal.getPoints()
    print("completed gradient extremals.")

    if DEFAULT_PLOT:
        learned_potential.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$', contour=True, contourCount=20)
        
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
        # the below lines color the starting point in purple 
        # and the final point in red to check direction.
        # plt.scatter(list(zip(*gradient_extremal_points))[0][-1],
        #             list(zip(*gradient_extremal_points))[1][-1],
        #             color="red", s=12, zorder=38)
        # plt.scatter(list(zip(*gradient_extremal_points))[0][0],
        #             list(zip(*gradient_extremal_points))[1][0],
        #             color="purple", s=12, zorder=38)

        ax = plt.gca()
        loc = plticker.MultipleLocator(base=0.001) # this locator puts ticks at regular intervals
        ax.axes.xaxis.set_major_locator(loc)
        ax.axes.yaxis.set_major_locator(loc)
        plt.xlabel(r"$u$")
        plt.ylabel(r"$v$")
        plt.tight_layout()
        if DEFAULT_SAVE:
            plt.savefig(f"{i}_2D.png")
        if DEFAULT_SHOWPLOT: plt.show()
        plt.close()

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
            mlab.savefig(f"{i}_3D.png")
        if DEFAULT_SHOWPLOT: mlab.show()
        mlab.close(all=True)

    initial_3D = gradient_extremal_3D[-2]
    if jnp.isnan(initial_3D[0]): 
        index = find_first_nan(jnp.array(gradient_extremal_3D)[:,0])
        initial_3D = jnp.array(gradient_extremal_3D)[index-1, :]+jnp.array([-0.01, 0.01, 0.01])
        logger.debug("There were NAN's in the continuation (likely infinite jacobian).")

    logger.info(f"NEW POINT: {initial_3D}")
    logger.info(f"CRITERIA TO END EARLY: {jnp.linalg.norm(jax.jacobian(mb.E)(initial_3D))}")

    jacobians = jax.vmap(jax.jacobian(mb.E))(jnp.array(gradient_extremal_3D))
    norms = jax.vmap(jnp.linalg.norm)(jacobians)
    if i > DEFAULT_MINCHARTSEXPECTED and jnp.min(norms) < DEFAULT_TOLERANCE:
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
