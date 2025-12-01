"""
Additional priors defined on top of the normal default priors in jim
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from typing import Callable, Union
from jimgw.core.prior import Prior

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from ninjax.pipes.pipe_utils import logger

# @jaxtyped
class NFPrior(Prior):
    
    nf: Transformed
    
    def __repr__(self):
        return f"NFPrior()"

    def __init__(
        self,
        nf_path: str,
        parameter_names: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(parameter_names, transforms)
        
        # TODO: this is just the structured I used but we should generalize somehow...
        # Define the PyTree structure for deserialization
        shape = (40_000, 4)
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key, 2)
        
        like_flow = block_neural_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros(shape[1])),
            nn_depth=5,
            nn_block_dim=8,
        )
        
        # Load the normalizing flow
        logger.info("Initializing the NF prior: deserializing leaves")
        _nf: Transformed = eqx.tree_deserialise_leaves(nf_path, like=like_flow)
        logger.info("Initializing the NF prior: deserializing leaves DONE")
        self.nf = _nf


    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from an NF.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        # Use the old-style PRNG key to get a seed
        seed = jax.random.uniform(rng_key, (1,)).astype(jnp.int32).at[0].get()
        rng_key = jax.random.key(seed)
        
        # Then use the seed to sample
        samples = self.nf.sample(rng_key, (n_samples, ))
        samples = samples.T
        
        _m_1, _m_2, lambda_1, lambda_2 = samples[0], samples[1], samples[2], samples[3]
        
        # Ensure m1 > m2
        m_1 = jnp.maximum(_m_1, _m_2)
        m_2 = jnp.minimum(_m_1, _m_2)
        
        # Ensure lambda1 > lambda2
        lambda_1 = jnp.minimum(lambda_1, lambda_2)
        lambda_2 = jnp.maximum(lambda_1, lambda_2)
        
        # Clip to avoid negative values
        lambda_1 = jnp.clip(lambda_1, 0.1)
        lambda_2 = jnp.clip(lambda_2, 0.1)
        
        # Gather as a new samples array
        samples = jnp.array([m_1, m_2, lambda_1, lambda_2])
        
        return self.add_name(samples)

    def log_prob(self, x: dict[str, Array]) -> Float:
        x_array = jnp.array([x[name] for name in self.parameter_names]).T
        return self.nf.log_prob(x_array)