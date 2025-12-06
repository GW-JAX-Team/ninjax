"""
Additional priors defined on top of the normal default priors in jim

TODO: NEEDS TESTING - This file contains custom prior classes that need validation

      NFPrior class:
      - Uses flowjax for normalizing flow priors
      - Hardcoded structure (40_000, 4) - NOT GENERAL!
      - Only tested for specific use case (4D BNS parameters)
      - PRNG key conversion is hacky (old-style to new-style)

      ACTION REQUIRED:
      1. TEST NFPrior with different dimensional priors
      2. Make NFPrior structure configurable (not hardcoded)
      3. Add validation to ensure NF model matches prior dimension
      4. Test PRNG key conversion works correctly
      5. Add unit tests for sample() and log_prob() methods
      6. Document expected format of nf_path file
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
    """Normalizing flow prior using flowjax

    TODO: CRITICAL ISSUES:
          1. Hardcoded structure: shape = (40_000, 4) - only works for 4D problems!
          2. Hardcoded nn_depth=5, nn_block_dim=8 - should be configurable
          3. No validation that loaded NF matches parameter_names dimension
          4. PRNG key conversion may fail for some JAX versions
          5. Assumes specific parameter ordering: [m_1, m_2, lambda_1, lambda_2]
          6. min/max logic (lines 83-89) is hardcoded for this specific prior

          TESTING NEEDED:
          - Test with different dimensions (not just 4D)
          - Test with different parameter orderings
          - Test log_prob() actually works correctly
          - Test that transforms parameter is used correctly
    """
    
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

        # TODO: CRITICAL - This initialization is HARDCODED and NOT GENERAL!
        #       The shape=(40_000, 4) assumes 4 parameters and 40k samples
        #       The nn_depth=5, nn_block_dim=8 are arbitrary choices
        #
        #       REQUIRED CHANGES:
        #       1. Infer shape[1] from len(parameter_names)
        #       2. Make shape[0], nn_depth, nn_block_dim configurable via **kwargs
        #       3. Validate that the loaded model matches expected dimensions
        #       4. Raise informative error if dimensions mismatch
        #
        # Define the PyTree structure for deserialization
        shape = (40_000, 4)  # WARNING: HARDCODED - will fail for non-4D priors!
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key, 2)

        like_flow = block_neural_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros(shape[1])),
            nn_depth=5,  # WARNING: HARDCODED
            nn_block_dim=8,  # WARNING: HARDCODED
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
        # TODO: CRITICAL - PRNG key conversion is fragile and may break!
        #       This converts old-style to new-style JAX keys in a hacky way
        #       Consider using a more robust approach or documenting JAX version requirements
        # Use the old-style PRNG key to get a seed
        seed = jax.random.uniform(rng_key, (1,)).astype(jnp.int32).at[0].get()
        rng_key = jax.random.key(seed)

        # Then use the seed to sample
        samples = self.nf.sample(rng_key, (n_samples, ))
        samples = samples.T

        # TODO: CRITICAL - This unpacking ASSUMES 4 parameters in this EXACT order!
        #       It will FAIL for any other configuration
        #       Need to make this general based on parameter_names
        _m_1, _m_2, lambda_1, lambda_2 = samples[0], samples[1], samples[2], samples[3]

        # TODO: CRITICAL - These constraints are HARDCODED for BNS masses and tidal parameters
        #       They assume:
        #       1. First two parameters are masses (m1 > m2 convention)
        #       2. Last two parameters are lambdas (lambda1 < lambda2 for m1 > m2)
        #       This will break for any other prior structure!
        #
        #       NEEDS: Generic constraint handling or removal if NF already learns constraints
        # Ensure m1 > m2
        m_1 = jnp.maximum(_m_1, _m_2)
        m_2 = jnp.minimum(_m_1, _m_2)

        # Ensure lambda1 < lambda2 (follows m1 > m2 convention)
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