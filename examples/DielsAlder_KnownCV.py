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

from mayavi import mlab

from scipy.linalg import eigvalsh
from scipy.spatial import ConvexHull

from gradient_extremals_on_manifolds.Continuation import Continuation
from gradient_extremals_on_manifolds.DiffGeom_Potential import DiffGeom_Potential
from gradient_extremals_on_manifolds.DiffusionMapCoordinates import DiffusionMapCoordinates
from gradient_extremals_on_manifolds.gaussian_process import make_gaussian_process
from gradient_extremals_on_manifolds.Sampler import Sampler


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


@jax.jit
def get_collective_variables(X):
    # index [1,:] to [9:,]
    X = jnp.reshape(X, (16,3), order='C')
    c1 = jnp.sqrt((X[1,0]-X[9,0])**2 + (X[1,1]-X[9,1])**2 + (X[1,2]-X[9,2])**2)
    # index [0,:], to [6:,]
    c2 = jnp.sqrt((X[0,0]-X[6,0])**2 + (X[0,1]-X[6,1])**2 + (X[0,2]-X[6,2])**2)
    return jnp.array([c1/10, c2/10])

class DataSample():

    def __init__(self, name, keep = 0.02, debug_plot = 0):

        self.COLVAR = f"data/{name}_COLVAR"
        self.FES = f"data/{name}_fes.dat"
        self.X_FILE = f"data/X-{name}.npy"
        self.KEEP = keep
        self.DEBUG_PLOT = debug_plot

        self.df_cv_energy = self.__gather_data_cv_energy()
        self.X = self.__gather_X()

        print("Thinning data.")
        self.kept_indices = self.__select_indices()
        self.df_cv_energy_reduced = self.df_cv_energy.loc[self.kept_indices]
        X_red = copy.deepcopy(self.X[jnp.array(list(self.kept_indices))])

        print("Processing data.")
        X_red = jnp.reshape(X_red, (X_red.shape[0],16,3), order='C')
        X_red = self.preprocess(X_red)
        self.X_reduced =  jnp.reshape(X_red, (X_red.shape[0],16*3), order='C')

        print("Data sample built.")

    def __gather_data_cv_energy(self):
        df = pd.read_csv(self.COLVAR+"_FORMATTED", 
                         names=["blank", "time", "C1", "C2"],
                         delimiter=" ", 
                         index_col=False, 
                         header=0)
        df.drop(index=df.index[-1],axis=0,inplace=True)
        fes =  pd.read_csv(self.FES, delimiter=",")
        colvars = jnp.array([fes["d1(A)"], fes["d2(A)"]]).T
        energy = jnp.array(fes["Free Energy(J)"]).T.reshape(-1,1)
        gpr = make_gaussian_process(colvars, energy, sigma=1e-1)
        data_original = jnp.array([df["C1"], df["C2"]]).T
        predicted_energy = jax.vmap(gpr)(data_original)
        df["Predicted Energy"] = predicted_energy[:,0]

        if self.DEBUG_PLOT == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('FES vs recreation')
            a = ax1.scatter(fes["d1(A)"], fes["d2(A)"], s=0.5, c=energy, vmin= -150, vmax = 100)
            plt.colorbar(a)
            a = ax2.scatter(df["C1"], df["C2"], s=0.5, c=df["Predicted Energy"], vmin= -150, vmax = 100)
            plt.colorbar(a)
            plt.tight_layout()
            plt.show()

        return df

    # for thinning out data
    def __select_indices(self):
        N = int(len(self.df_cv_energy["C1"])*self.KEEP)
        indices_to_keep = random.sample(range(0, len(self.df_cv_energy.index)), N)
        indices_to_keep.sort()
        return indices_to_keep

    def preprocess(self, x, fixed_atom=None): # rmsd + kabsh
        processed= np.array(x)
        original = np.copy(x)
        for i in range(0,x.shape[0]):
            if fixed_atom == None: 
                original[i,:,:] -= rmsd.centroid(original[i,:,:]) # now x,y,z is centered at 0,0,0
            else:
                original[i,:,:] -= original[i,fixed_atom,:] # center fixed atom at 0, 0, 0
            U = rmsd.kabsch(x[i,:,:], original[0,:,:]) # understand rotation matrix wrt 1st frame which is "base"
            processed[i,:,:] = (jnp.dot(original[i,:,:],U)) # finally rotate x,y,z to be in line
        return processed

    def __gather_X(self):
        X = jnp.load(self.X_FILE, mmap_mode='r')
        return X

# grab sample
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

data_sample = DataSample(name="5_7-3_5", keep=0.02, debug_plot=1)

# plot thinning as fyi
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Thinning')
ax1.scatter(data_sample.df_cv_energy["C1"], data_sample.df_cv_energy["C2"], s=0.5, c=data_sample.df_cv_energy["Predicted Energy"], vmin= -150, vmax = 100)
a = ax2.scatter(data_sample.df_cv_energy_reduced["C1"], data_sample.df_cv_energy_reduced["C2"], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= -150, vmax = 100)
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
a = ax1.scatter(Y_reduced[:,0], Y_reduced[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= -150, vmax = 100)
ax1.set_title("colored by true energy")
ax1.set_aspect("equal")
plt.colorbar(a)
a = ax2.scatter(w[:,0], w[:,1], s=0.5, c=jax.vmap(learned_potential_func)(w), vmin= -150, vmax = 100)
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

initial_point = data_sample.X_reduced[np.argmin(energies),:]
initial_point_coordinates = get_collective_variables(initial_point)
initial_point_coordinates = jnp.array([0.505, 0.469])

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
                    maxiter = 200,
                    verbose = DEFAULT_VERBOSE,
                    tolerance = 0.01,
                    h = 0.0005,
                    max_cond = lambda x:learned_potential.potential_func.get_variance(x) > 5E-2)

print("Starting continuation.")
path.start()
path_points = path.getPoints()

print("Finishing continuation.")

print("This was the final point: ", jnp.array(path_points)[-1,:])

# plot mesh for checking V
fig, (ax1, ax2) = plt.subplots(1, 2)
w1 = np.linspace(min(Y_reduced[:,0]), max(Y_reduced[:,0]), 100)
w2 = np.linspace(min(Y_reduced[:,1]), max(Y_reduced[:,1]), 100)
w1v, w2v = np.meshgrid(w1, w2)
w = jnp.vstack([w1v.flatten(),w2v.flatten()]).T;
fig.suptitle('Learned Potential Function')
a = ax1.scatter(Y_reduced[:,0], Y_reduced[:,1], s=0.5, c=data_sample.df_cv_energy_reduced["Predicted Energy"], vmin= -150, vmax = 100)
ax1.set_title("colored by true energy")
ax1.set_aspect("equal")
plt.colorbar(a)
a = ax2.scatter(w[:,0], w[:,1], s=0.5, c=jax.vmap(learned_potential_func)(w), vmin= -150, vmax = 100)
ax2.plot(list(zip(*path_points))[0],
         list(zip(*path_points))[1], color="orange")
ax2.scatter(initial_point_coordinates[0], initial_point_coordinates[1], s=10.0, c="red")
ax2.set_aspect("equal")
ax2.set_title("colored by pullback energy")
plt.colorbar(a)
plt.tight_layout()
plt.show()

# mb.plot_color_mesh(colorbarTitle=r'$Z=\psi^\star E$', vrangeOn=True)
# plt.plot(list(zip(*path_points))[0],
#          list(zip(*path_points))[1], color="orange")
# plt.xlabel(r"$u$")
# plt.ylabel(r"$v$")

# plt.show()

