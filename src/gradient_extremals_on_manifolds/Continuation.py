#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import svd


class Continuation:

    def __init__(self,
                 initial_point,
                 functions, # array of functions
                 maxiter = 100,
                 h = 0.05,
                 verbose = 0, #1, #2, the larger the more error messages
                 tolerance = 0.001,
                 max_cond = lambda x:0):

        self.functions = functions
        self.function = functions[0]
        self.maxiter = maxiter
        self.verbose = verbose
        self.tolerance = tolerance
        self.initial_point = initial_point
        self.all_points=[]
        self.h = h
        self.h_max = h
        self.max_cond = max_cond

    # equation that contains [f(x,y)=0, tangent.T(z-z0)-h=0]
    @partial(jit, static_argnums=(0,))
    def F(self, z, z0, tan):
        main_equations = jnp.array([function(z) for function in self.functions])
        return jnp.append(main_equations, jnp.dot(tan, z-z0)-self.h)

    @partial(jit, static_argnums=(0,))
    def F_jacobian(self, z, z0, tan):
        main_equations = jnp.array([jax.jacobian(function)(z) for function in self.functions])
        jacob = jnp.append(main_equations, jnp.expand_dims(tan, axis=0), axis=0)
        if self.verbose > 2: jax.debug.print("jacobian_F: {x}", x=jacob)
        return jacob

    # corrector step
    def newton_raphson(self, z, z0, maxiter, tan):
        if self.verbose > 1: print("iter: 0, starting z:", z)
        for i in range(maxiter):
            soln = self.F(z, z0, tan)
            if self.verbose > 1:
                print("Current solution: ", soln, " Norm: ", jnp.linalg.norm(soln))
            if jnp.linalg.norm(soln) < self.tolerance:
                if self.verbose > 1: print("Found value: ", z)
                return z
            if self.verbose > 2:
                print("Grad at : ", z, " is ", jax.grad(self.function)(z),
                      " & the Jacobian: ", (self.F_jacobian(z, z0, tan)))
            z = z + jnp.linalg.solve(self.F_jacobian(z, z0, tan), -self.F(z, z0, tan))
            if self.verbose > 1: print("iter: %d, new:" % (i+1), z)
        if self.verbose > 2:
            print("Z: ", z, " and Norm of last solution when correcting: ", jnp.linalg.norm(soln))
            z = z + jnp.linalg.solve(self.F_jacobian(z, z0, tan), -self.F(z, z0, tan))
            soln = self.F(z, z0, tan)
            print("z: ", z, "Check to see if norm is getting smaller: ", jnp.linalg.norm(soln))
            print("Max iterations reached. No solution found.")
        if self.verbose > 0:
            print("Newton_Raphson did not converge. Norm of solution: ", jnp.linalg.norm(soln))
            print("The current zeros of the equations: ", soln)
            print("Which is given by z = ", z)
            print("With this jacobian: ", self.F_jacobian(z, z0, tan))
        return z

    def nullspace(self, A, atol=1e-13, rtol=0):
        A = jnp.atleast_2d(A)
        U, s, V = svd(A)
        return jnp.squeeze(jnp.asarray(V[:, len(s)]))

    def tangent(self, z):
        main_equations = jnp.array([jax.jacobian(function)(z) for function in self.functions])
        return self.nullspace(main_equations)

    # predictor step
    def compute_step(self, z0, old_tangent):
        tan = self.tangent(z0) # obtain tangent to original point
        if self.verbose > 1: print("tangent: ", tan)
        if jnp.dot(old_tangent, tan) < 0: # adaptively reverses direction
            tan = -tan
            if self.verbose > 1: print("Reversing sign of tangent")

        for i in range(20):
            z_new = z0 + self.h*tan # predict new step along tangent
            if self.verbose > 1:
                print("Predicted: ", z_new)
                print("post prediction: ", z_new)
            z_new = self.newton_raphson(z_new, z0, 10, tan)
            if z_new is not None:
                if self.h < self.h_max:
                    self.h = min(1.2 * self.h, self.h_max)
                break
            self.h = 0.5 * self.h
            if self.verbose > 0: print("Adaptive step sizing! h is now: ", self.h)

        if self.verbose > 1:
            print("post correction: ", z_new)
        return z_new, tan

    def start(self):
        z = self.initial_point
        self.all_points = []
        z = self.newton_raphson(z, z, 100, self.tangent(z))

        self.all_points.append(z)
        tan = -self.tangent(z)*-1
        for i in range(self.maxiter):
            if self.verbose > 0:
                print("iteration: ", i)
                print("max_cond: ", self.max_cond(z))
            z, tan = self.compute_step(z, tan)
            if self.verbose > 0:
                print("FOUND POINT: ", z)
            self.all_points.append(z)
            if self.max_cond(z):
                if self.verbose > 0:
                    print("REACHED TOLERANCE CONDITION")
                    print("iter no: ", print(i))
                break
        return None

    def getPoints(self):
        return self.all_points
