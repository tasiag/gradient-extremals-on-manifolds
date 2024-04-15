'''
Full implementation of algorithm (sampling, manifold learning, differential geometry)
for tracing gradient extremals Diels Alder reaction.


THIS IS BASED IN RMSD DMAPS.

'''

import jax
import jax.numpy as jnp

import numpy as np

import copy

import logging

from SphericalMB import SphericalMB # this can't be removed otherwise plotting doesn't work...
from utils import plot_spherical_potential, plot_points3d, plot_lines3d, \
                  setup_log, get_direction, find_first_nan

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import math
import pandas as pd
import rmsd
import random
import scipy

from ase import Atoms

from mayavi import mlab

from scipy.linalg import eigvalsh
from scipy.spatial import ConvexHull

from gradient_extremals_on_manifolds.Continuation import Continuation
from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential
from gradient_extremals_on_manifolds.DiffusionMapCoordinates import DiffusionMapCoordinates
from gradient_extremals_on_manifolds.gaussian_process import make_gaussian_process
from gradient_extremals_on_manifolds.Sampler import Sampler
from gradient_extremals_on_manifolds.distance import RMSDDistance

from DataSample import DataSample, get_collective_variables

# Options (True or False)
DEFAULT_VERBOSE = True # False for INFO only, True for DEBUG
DEFAULT_PLOT = True
DEFAULT_SHOWPLOT = False
DEFAULT_SAVE = True

# Problem-specific parameters to set
DEFAULT_ITERATIONS = 30
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_TOLERANCE = 10
DEFAULT_THRESHOLD_SHRINKSTEPSIZE = 150 # shrink step size when approach saddle
DEFAULT_MINCHARTSEXPECTED = 4
DEFAULT_SEED = 1234#3452

plt.close()

# Text formatting
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.family' : 'lmodern',
          'font.size' : 18
          }
plt.rcParams.update(params) 

name = "STC_US_2.5_2.9" #"STC_US_2.7_3.0"# "2.5_5.5-2.5_5.5" #"US_4_4" "4.5_7.5-4.5_7.5"
data_sample = DataSample(name=name, keep=0.1, debug_plot=1, u=True, sigma=.1, seed=DEFAULT_SEED)
vmin, vmax = data_sample.vmin, data_sample.vmax


# plot thinning as fyi
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Thinning')
vmin, vmax = data_sample.vmin, data_sample.vmax
ax1.scatter(data_sample.df_cv_energy["C1"], data_sample.df_cv_energy["C2"], s=0.5, c=data_sample.df_cv_energy["Predicted Energy"], vmin= vmin, vmax = vmax)
a = ax2.scatter(data_sample.df_cv_energy_reduced["C1"], data_sample.df_cv_energy_reduced["C2"], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= vmin, vmax = vmax)
plt.colorbar(a)
plt.tight_layout()
plt.show()


print("Gathering reduced coordinates.")
# gather reduced coordinates
Y_reduced = jax.vmap(get_collective_variables)(data_sample.X_reduced)

phi = DiffusionMapCoordinates(16*3, 2, RMSDDistance)
print(data_sample.X_reduced_byatoms_NOTPROCESSED.shape)
dmaps = phi.learn(data_sample.X_reduced_byatoms_NOTPROCESSED)
# plot learned dmaps as fyi

psi = make_gaussian_process(dmaps, data_sample.X_reduced)

status = phi.save_gpr("ForBryan/")
status = data_sample.save_samples_as_xyz("ForBryan/samples.xyz")

print("SHAPE dmaps: ", dmaps.shape)
energies = jnp.array(data_sample.df_cv_energy_reduced["Predicted Energy"]).reshape(-1, 1)
print("ENERGIES = ", energies.shape)
learned_potential_func = make_gaussian_process(dmaps, energies, sigma=5E-2)

# plot mesh for checking V
fig, (ax1, ax2) = plt.subplots(1, 2)
w1 = np.linspace(min(dmaps[:,0]), max(dmaps[:,0]), 100)
w2 = np.linspace(min(dmaps[:,1]), max(dmaps[:,1]), 100)
w1v, w2v = np.meshgrid(w1, w2)
w = jnp.vstack([w1v.flatten(),w2v.flatten()]).T;
fig.suptitle('Learned Potential Function')
a = ax1.scatter(dmaps[:,0], dmaps[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= vmin, vmax = vmax)
ax1.set_title("colored by true energy")
ax1.set_aspect("equal")
plt.colorbar(a)
a = ax2.scatter(w[:,0], w[:,1], s=0.5, c=jax.vmap(learned_potential_func)(w), vmin= vmin, vmax = vmax)
ax2.set_aspect("equal")
ax2.set_title("colored by pullback energy")
plt.colorbar(a)
plt.tight_layout()
plt.show()

print("Learning potential function.")

# learn potential system
learned_potential = DiffGeom_Potential(learned_potential_func, 
                                       phi,
                                       psi)

print("Learned DiffGeom_Potential.")

initial_point = data_sample.X_reduced_byatoms_NOTPROCESSED[np.argmin(energies),:]
#initial_point = data_sample.X_reduced[np.argmin(energies),:]

print(initial_point)
print(initial_point.shape)

initial_point_coordinates = phi(initial_point)

#initial_point_coordinates = jnp.array([0.27, 0.30])
print("INITIAL POINT COORDINATES: ", initial_point_coordinates)


# ### now that we know the components we can start the process 

path = Continuation(initial_point=initial_point_coordinates,
                    functions = [learned_potential.equal_nullcline],
                    maxiter = 1000,
                    verbose = DEFAULT_VERBOSE,
                    tolerance = 0.000001,
                    h = -0.000001,#0.0005,
                    max_cond = lambda x:learned_potential.potential_func.get_variance(x) > 1E-4)#5E-2)

print("Starting continuation.")
path.start()
path_points = path.getPoints()

path2 = Continuation(initial_point=initial_point_coordinates,
                    functions = [learned_potential.equal_nullcline],
                    maxiter = 1000,
                    verbose = DEFAULT_VERBOSE,
                    tolerance = 0.000001,
                    h = 0.000001,#0.0005,
                    max_cond = lambda x:learned_potential.potential_func.get_variance(x) > 1E-4)#5E-2)

print("Starting continuation.")
path2.start()
path_points2 = path2.getPoints()

print("Finishing continuation.")

print("This was the final point: ", jnp.array(path_points)[-1,:])

# plot mesh for checking V
fig, (ax1, ax2) = plt.subplots(1, 2)
w1 = np.linspace(min(dmaps[:,0]), max(dmaps[:,0]), 100)
w2 = np.linspace(min(dmaps[:,1]), max(dmaps[:,1]), 100)
w1v, w2v = np.meshgrid(w1, w2)
w = jnp.vstack([w1v.flatten(),w2v.flatten()]).T;
fig.suptitle('Learned Potential Function')
a = ax1.scatter(dmaps[:,0], dmaps[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= vmin, vmax = vmax)
ax1.set_title("colored by true energy")
ax1.set_aspect("equal")
plt.colorbar(a)
a = ax2.scatter(w[:,0], w[:,1], s=0.5, c=jax.vmap(learned_potential_func)(w), vmin= vmin, vmax = vmax)
ax2.plot(list(zip(*path_points))[0],
         list(zip(*path_points))[1], color="orange")
ax2.scatter(list(zip(*path_points))[0],
         list(zip(*path_points))[1], color="orange", s=0.5)
ax2.plot(list(zip(*path_points2))[0],
         list(zip(*path_points2))[1], color="orangered")
ax2.scatter(list(zip(*path_points2))[0],
         list(zip(*path_points2))[1], color="orangered", s=0.5)
ax2.scatter(initial_point_coordinates[0], initial_point_coordinates[1], s=10.0, c="red")
ax2.set_aspect("equal")
ax2.set_title("colored by pullback energy")
plt.colorbar(a)
plt.tight_layout()
plt.show()

print("ENDING ORANGE: ", path_points[-1])
print ("ENDING ORANGERED: ", path_points2[-1])

np.savetxt("ForBryan/new_start.csv",
            np.array([path_points[-1], path_points2[-1]]),
            delimiter =", ")

print("NORM ENDING ORANGE: ", np.linalg.norm(learned_potential.grad(np.array(path_points[-1]))))
print("NORM ENDING ORANGERED: ", np.linalg.norm(learned_potential.grad(np.array(path_points[-2]))))

# # mb.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$', vrangeOn=True)
# # plt.plot(list(zip(*path_points))[0],
# #          list(zip(*path_points))[1], color="orange")
# # plt.xlabel(r"$u$")
# # plt.ylabel(r"$v$")

# # plt.show()

