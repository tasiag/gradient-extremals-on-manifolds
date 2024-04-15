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

@jax.jit
def get_collective_variables(X):
    # index [1,:] to [9:,]
    X = jnp.reshape(X, (16,3), order='C')
    c1 = jnp.sqrt((X[1,0]-X[9,0])**2 + (X[1,1]-X[9,1])**2 + (X[1,2]-X[9,2])**2)
    # index [0,:], to [6:,]
    c2 = jnp.sqrt((X[0,0]-X[6,0])**2 + (X[0,1]-X[6,1])**2 + (X[0,2]-X[6,2])**2)
    return jnp.array([c1/10, c2/10])

class DataSample():

    def __init__(self, name, keep = 0.02, debug_plot = 0, u=False, sigma=None, seed = None):

        # self.COLVAR = f"data/{name}_COLVAR"
        self.FES = f"data/{name[0]}_fes.dat"#_smoothed.dat"
        self.X_FILE = f"data/{name[0]}_X_{name[1]}.npy"
        self.KEEP = keep
        self.DEBUG_PLOT = debug_plot
        self.seed = seed

        if name[0] == "U" or u:
            self.scaler = 1#10
            self.vmin = 0#0
            self.vmax = 15#15
            self.sigma = 1E-1
            if sigma is not None:
                self.sigma = sigma#1E-2#0.009
        else:
            self.scaler = 1
            self.vmin = -150
            self.vmax = 100
            self.sigma = 1E-1

        self.reference_frame = None

        # get X, calculate true colvars, get df_cv_energy using fes
        self.X = self.__gather_X()
        self.colvars = jax.vmap(get_collective_variables)(self.X)
        print(self.colvars[0,:], self.colvars[-1,:])
        # self.df_cv_energy = self.__gather_data_cv_energy()


        print("Thinning data.")
        self.kept_indices = self.__select_indices()
        # self.df_cv_energy_reduced = self.df_cv_energy.loc[self.kept_indices]
        X_red = copy.deepcopy(self.X[jnp.array(list(self.kept_indices))])
        
        print("Processing data.")
        X_red = jnp.reshape(X_red, (X_red.shape[0],16,3), order='C')
        self.X_reduced_NOTPROCESSED = jnp.reshape(X_red, (X_red.shape[0],16*3), order='C')
        self.X_reduced_byatoms_NOTPROCESSED = X_red
        X_red = self.preprocess(X_red)
        self.X_reduced_byatoms = X_red
        self.X_reduced =  jnp.reshape(X_red, (X_red.shape[0],16*3), order='C')

        self.colvars_reduced = self.colvars[self.kept_indices,:]

        print("Data sample built.")

    def __gather_data_cv_energy(self):
        fes =  pd.read_csv(self.FES, delimiter=",")

        df = {"C1": self.colvars[:,0], 
              "C2": self.colvars[:,1]}

        # understand if there's a difference in scale
        magnitude = round(np.mean(fes["d1(A)"])/np.mean(df["C1"]), -1)

        colvars = jnp.array([fes["d1(A)"]/magnitude, fes["d2(A)"]/magnitude]).T
        energy = jnp.array(fes["Free Energy(J)"]).T.reshape(-1,1)
        print("ORIGINAL: ", np.mean(energy))
        # check sigmas!!
        # originally had 1e-1, .009
        gpr = make_gaussian_process(colvars, self.scaler*energy, sigma=self.sigma)
        data_original = jnp.array([df["C1"], df["C2"]]).T
        predicted_energy = jax.vmap(gpr)(data_original)
        print("PREDICTED: ", np.mean(predicted_energy))
        df["Predicted Energy"] = predicted_energy[:,0]


        if self.DEBUG_PLOT == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('FES vs recreation')
            a = ax1.scatter(fes["d1(A)"]/10, fes["d2(A)"]/magnitude, s=0.5, c=energy, vmin= self.vmin, vmax = self.vmax)
            plt.colorbar(a)
            a = ax2.scatter(df["C1"], df["C2"], s=0.5, c=df["Predicted Energy"], vmin= self.vmin, vmax = self.vmax)
            plt.colorbar(a)
            plt.tight_layout()
            plt.show()

        return df

    # for thinning out data
    def __select_indices(self):
        if self.seed is not None:
            random.seed(self.seed)
        N = int(len(self.colvars[:,0])*self.KEEP)
        indices_to_keep = random.sample(range(0, len(self.colvars[:,0])), N)
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
        self.reference = original[0,:,:]
        return processed

    def get_reference_point(self):
        return self.reference

    def convert_to_xyz(self, name, symbols, positions):
        try:
            atoms = Atoms(symbols=symbols, positions=positions)
            atoms.write(name)
            return True
        except Exception as e:
            print(f"Could not save xyz: {e}")
            return False

    def __gather_X(self):
        X = jnp.load(self.X_FILE, mmap_mode='r')
        return X

    def save_samples(self, name):

        print(f"!!!!!!! saving samples, and they have shape: {self.X_reduced.shape}")
        try:
            np.savetxt(name, self.X_reduced, delimiter=",")
            return True
        except Exception as e:
            print(f"Could not save samples: {e}")
            return False


    def save_samples_as_xyz(self, name):
        print(f"!!!!!!! saving samples, and they have shape: {self.X_reduced_byatoms.shape}")

        def write_xyz(atoms, filename, mode='a'):
            with open(filename, mode) as f:
                f.write("\n")
                f.write(f"{len(atoms)}\n")
                f.write('Properties=species:S:1:pos:R:3 pbc="F F F"\n')
                for atom in atoms:
                    f.write(f"{atom.symbol} {atom.x} {atom.y} {atom.z}\n")

        success = True
        for point in self.X_reduced_byatoms:
            try:
                atoms = Atoms(symbols=['C','C','H','H','H','H','C','C','C','C','H','H','H','H','H','H'],
                              positions=point)
                write_xyz(atoms, name)
            except Exception as e:
                print(f"Could not save xyz: {e}")
                success = False
        return success

if __name__ == "__main__":
    pass