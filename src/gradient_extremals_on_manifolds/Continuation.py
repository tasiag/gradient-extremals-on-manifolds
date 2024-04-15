#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import svd

import logging
import sys


class Continuation:

    def setup_log(self, name, verbose):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler = logging.FileHandler(f"./{name}.log")
        if verbose:
            log_handler.setLevel(logging.DEBUG)
        else:
            log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(log_format)

        logger.addHandler(log_handler)
        logger.addHandler(stream_handler)
        return logger 

    def __init__(self,
                 initial_point,
                 functions, # array of functions
                 maxiter = 100,
                 h = 0.05,
                 verbose = 0, #0 prints info only, 1 prints debug level
                 tolerance = 0.001,
                 max_cond = lambda x:0):

        self.logger = self.setup_log("continuation", verbose)
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
        return jacob

    # corrector step
    def newton_raphson(self, z, z0, maxiter, tan):
        for i in range(maxiter):
            soln = self.F(z, z0, tan)
            self.logger.info(f"Newton-Raphson | iter: {i} | z: {z} | soln: {soln} | norm: {jnp.linalg.norm(soln)}")
            if jnp.linalg.norm(soln) < self.tolerance:
                self.logger.debug(f"Found value: {z}")
                return z
            self.logger.debug(f"Newton-Raphson | Grad at {z} is {jax.grad(self.function)(z)} \
                | Jacobian {self.F_jacobian(z, z0, tan)}")
            z = z + jnp.linalg.solve(self.F_jacobian(z, z0, tan), -self.F(z, z0, tan))
        self.logger.debug("Newton-Raphson | Max iterations reached. No solution found.")
        self.logger.debug(f"Newton-Raphson | z: {z} norm: {jnp.linalg.norm(soln)}")
        self.logger.info(f"Newton-Raphson | The current zeros of the equations: {soln}")
        self.logger.info(f"Newton-Raphson | With this jacobian: {self.F_jacobian(z, z0, tan)}")
        return None

    def nullspace(self, A, atol=1e-13, rtol=0):
        A = jnp.atleast_2d(A)
        U, s, V = svd(A)
        return jnp.squeeze(jnp.asarray(V[:, len(s)]))

    def tangent(self, z):
        main_equations = jnp.array([jax.jacobian(function)(z) for function in self.functions])
        return self.nullspace(main_equations)

    def compute_step(self, z0, old_tangent):
        tan = self.tangent(z0) # obtain tangent to original point
        self.logger.info(f"tangent: {tan}")
        if jnp.dot(old_tangent, tan) < 0: # adaptively reverses direction
            tan = -tan
            self.logger.info("Predictor | Reversing sign of tangent")

        for i in range(20):
            z_new = z0 + self.h*tan # predict new step along tangent
            self.logger.info(f"Predictor | Predicted Solution: {z_new}")
            z_new = self.newton_raphson(z_new, z0, 100, tan)
            if z_new is not None: # Newton-Raphson converged, step h back up if needed
                if self.h < self.h_max:
                    self.h = min(1.2 * self.h, self.h_max)
                break
            if z_new is None: 
                self.logger.debug(f"z_new is none at compute_step's for loop, iteration {i}")
            self.h = 0.5 * self.h # Newton-Raphson did not converge, reduce step size
            self.logger.info(f"Corrector | Adaptive step sizing! h is now: {self.h} at iteration {i}")

        self.logger.info(f"Corrector | Corrected solution: {z_new}")
        return z_new, tan

    def start(self):
        z = self.initial_point
        self.all_points = []

        z = self.newton_raphson(z, z, 1000, self.tangent(z))
        if z is None: 
            self.logger.error("Continuation | Initial starting point did not converge.")
            return None

        self.all_points.append(z)
        tan = self.tangent(z)
        for i in range(self.maxiter):
            self.logger.info(f"Continuation | iteration: {i} max condition: {self.max_cond(z)}")
            z, tan = self.compute_step(z, tan)
            if z is None:
                self.logger.warning("Continuation | Terminating early. Could not meet supplied tolerance.")
                break
            if self.max_cond(z):
                self.logger.info(f"Continuation | Reached max condition, iteration {i}")
                break
            self.all_points.append(z)

        return jnp.array([self.all_points[-1][1]-self.all_points[-2][1], self.all_points[-1][0]-self.all_points[-2][0]])

    def getPoints(self):
        return self.all_points
