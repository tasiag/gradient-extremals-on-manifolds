import jax
import jax.numpy as jnp

class Sampler:

    def __init__(self,
                 potential_ambient,
                 constraint, # written such that it is f(x)=0
                 timestep = 1E-8,
                 mu = 100000,
                 no_iters = 500,
                 noise_level = 0.05):
        self.V = potential_ambient
        self.constraint = constraint
        self.timestep = timestep
        self.mu = mu
        self.noise_level = noise_level
        self.no_iters = no_iters

    def _generate_candidates(self, initial, N, key = jax.random.PRNGKey(0)):
        noise = jax.random.uniform(key=key,
            minval=-self.noise_level, maxval=self.noise_level,
            shape=(N, int(initial.shape[0])))
        return initial + noise

    def draw_samples(self, initial, N, key = jax.random.PRNGKey(0)):
        candidates = self._generate_candidates(initial, N, key)
        for i in range(self.no_iters):
            constraint = jnp.expand_dims(jax.vmap(self.constraint)(candidates), axis=1)
            grad_constraint = jax.vmap(jax.jacobian(self.constraint))(candidates)
            df = jax.vmap(jax.jacobian(self.V))(candidates)
            constraint_portion = self.mu*constraint*grad_constraint
            candidates = candidates - (constraint_portion + df) * self.timestep
        return candidates
