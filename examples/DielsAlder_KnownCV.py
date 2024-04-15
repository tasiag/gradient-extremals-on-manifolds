'''
Full implementation of algorithm (sampling, manifold learning, differential geometry)
for tracing gradient extremals on the Muller Brown Potential mapped onto a sphere
by using sampled point clouds.
'''

import jax
import jax.numpy as jnp

import numpy as np

import copy

import logging

from SphericalMB import SphericalMB
from utils import plot_spherical_potential, plot_points3d, plot_lines3d, \
                  setup_log, get_direction, find_first_nan

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import math
import pandas as pd
import rmsd
import random
import scipy

from mayavi import mlab

from scipy.linalg import eigvalsh
from scipy.spatial import ConvexHull

from gradient_extremals_on_manifolds.Continuation import Continuation
from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential
from gradient_extremals_on_manifolds.DiffusionMapCoordinates import DiffusionMapCoordinates
from gradient_extremals_on_manifolds.gaussian_process import make_gaussian_process
from gradient_extremals_on_manifolds.Sampler import Sampler

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
DEFAULT_SEED = 2345#3452

plt.close()

# Text formatting
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.family' : 'lmodern',
          'font.size' : 18
          }
plt.rcParams.update(params) 

# grab samples OLD
# ##########
# 5_7-5_7: 
# initial_point = data_sample.X_reduced[np.argmin(energies),:]
# initial_point_coordinates = get_collective_variables(initial_point)
# tolerance:  > 1E-5
# h: h = -0.001,
# ##########
# 5_7-3_5:
# initial_point_coordinates = [0.4977341, 0.5872208]
# tolerance:  > 1E-4
# h: h = -0.0005,
# ##########
# 3_5-3_5:
# initial_point_coordinates = [0.4978457,  0.48870495]
# tolerance:  > 1E-3
# h: h = -0.0005,
# ##########
# 5_7-3_5:
# initial_point_coordinates = [0.505,  0.469]
# tolerance:  > 5E-3
# h: h = -0.0005,
# ##########

# grab samples with wider width
# ##########
# "4.5_7.5-4.5_7.5"
# initial_point = data_sample.X_reduced[np.argmin(energies),:]
# initial_point_coordinates = get_collective_variables(initial_point)
# initial_point_coordinates = 
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = +/- 0.001,
# ##########
# "2.5_5.5-4.5_7.5":
# initial_point_coordinates = [0.4398544,  0.51814544]
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = -0.001,
# ##########
# "2.5_5.5-2.5_5.5":
# initial_point_coordinates = [0.3212834, 0.4791346]
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = -0.001,
# ##########
# START TWO
# ##########
# "4.5_7.5-2.5_5.5"
# initial_point = data_sample.X_reduced[np.argmin(energies),:]
# initial_point_coordinates = get_collective_variables(initial_point)
# initial_point_coordinates = 
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = +/- 0.001,
# ##########
# "2.5_5.5-2.5_5.5":
# initial_point_coordinates = [0.46017015, 0.40840447]
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = -0.001,
# ##########

# STC US RUN
# ##########
# "STC_US_4_4"
# initial_point = data_sample.X_reduced[np.argmin(energies),:]
# initial_point_coordinates = get_collective_variables(initial_point)
# initial_point_coordinates = 
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = +/- 0.001,
# ##########
# "STC_US_3.3_3.5"
# initial_point_coordinates = 0.33, 0.35
# tolerance: 0.01 
# threshold: > 1E-5
# h: h = +/- 0.001,
# ##########
# "STC_US_2.9_3.4"
# initial_point_coordinates = 0.29, 0.34
# tolerance: 0.01 
# threshold: > 2E-4
# h: h = +/- 0.001,
# sigma: 0.1


name = "STC_US_2.5_2.9" #"STC_US_2.6_2.9" # "STC_US_2.7_3.0"
data_sample = DataSample(name=name, keep=0.2, debug_plot=1, u=True, sigma=.2)


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

# learn components 
print("Learning components.")
psi = make_gaussian_process(Y_reduced, data_sample.X_reduced)

print("SHAPE Y_REDUCED: ", Y_reduced.shape)
energies = jnp.array(data_sample.df_cv_energy_reduced["Predicted Energy"]).reshape(-1, 1)
print("ENERGIES = ", energies.shape)
learned_potential_func = make_gaussian_process(Y_reduced, energies)

# plot mesh for checking V
fig, (ax1, ax2) = plt.subplots(1, 2)
w1 = np.linspace(min(Y_reduced[:,0]), max(Y_reduced[:,0]), 100)
w2 = np.linspace(min(Y_reduced[:,1]), max(Y_reduced[:,1]), 100)
w1v, w2v = np.meshgrid(w1, w2)
w = jnp.vstack([w1v.flatten(),w2v.flatten()]).T;
fig.suptitle('Learned Potential Function')
a = ax1.scatter(Y_reduced[:,0], Y_reduced[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= vmin, vmax = vmax)
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
                                       get_collective_variables,
                                       psi)

print("Learned DiffGeom_Potential.")

#initial_point = data_sample.X_reduced[np.argmin(energies),:]
#initial_point_coordinates = get_collective_variables(initial_point)
initial_point_coordinates = jnp.array([0.25, 0.29])
print("INITIAL POINT COORDINATES: ", initial_point_coordinates)

# print(learned_potential.potential(initial_point_coordinates)) # pass 
# print(jax.jacobian(learned_potential.potential)(initial_point_coordinates)) # pass
# print(jax.jacobian(learned_potential.psi)(initial_point_coordinates))

# print("These are known to fail.")
# print(learned_potential.inverse_metric(initial_point_coordinates)) # fail
# print(learned_potential.y_nullcline(initial_point_coordinates)) # fail
# print(jax.jacobian(learned_potential.y_nullcline)(initial_point_coordinates)) # fail

# input("Now type Ctrl-C to exit.")

### now that we know the components we can start the process 

path = Continuation(initial_point=initial_point_coordinates,
                    functions = [learned_potential.equal_nullcline],
                    maxiter = 1000,
                    verbose = DEFAULT_VERBOSE,
                    tolerance = 0.01, #0.01,
                    h = -0.001,#0.0005,
                    max_cond = lambda x:learned_potential.potential_func.get_variance(x) > 1E-5)#5E-2)

print("Starting continuation.")
path.start()
path_points = path.getPoints()

path2 = Continuation(initial_point=initial_point_coordinates,
                    functions = [learned_potential.equal_nullcline],
                    maxiter = 1000,
                    verbose = DEFAULT_VERBOSE,
                    tolerance = 0.01, #0.01,
                    h = 0.001,#0.0005,
                    max_cond = lambda x:learned_potential.potential_func.get_variance(x) > 1E-5)#5E-2)

print("Starting continuation.")
path2.start()
path_points2 = path2.getPoints()

print("Finishing continuation.")

print("This was the final point: ", jnp.array(path_points)[-1,:])

# plot mesh for checking V
fig, (ax1, ax2) = plt.subplots(1, 2)
w1 = np.linspace(min(Y_reduced[:,0]), max(Y_reduced[:,0]), 100)
w2 = np.linspace(min(Y_reduced[:,1]), max(Y_reduced[:,1]), 100)
w1v, w2v = np.meshgrid(w1, w2)
w = jnp.vstack([w1v.flatten(),w2v.flatten()]).T;
fig.suptitle('Learned Potential Function')
a = ax1.scatter(Y_reduced[:,0], Y_reduced[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= vmin, vmax = vmax)
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

print("NORM ENDING ORANGE: ", np.linalg.norm(learned_potential.grad(np.array(path_points[-1]))))
print("NORM ENDING ORANGERED: ", np.linalg.norm(learned_potential.grad(np.array(path_points[-2]))))
# mb.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$', vrangeOn=True)
# plt.plot(list(zip(*path_points))[0],
#          list(zip(*path_points))[1], color="orange")
# plt.xlabel(r"$u$")
# plt.ylabel(r"$v$")

# plt.show()

