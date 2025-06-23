"""
Additional priors defined on top of the normal default priors in jim
"""

import jax
import jax.numpy as jnp
import json
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from typing import Callable, Union
from jimgw.prior import Prior, Uniform, PowerLaw, Composite

import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.distributions import Normal, Transformed

from ninjax.pipes.pipe_utils import logger
from ninjax.transforms import Mc_q_H0_dL_to_m1_m2_source, Mc_q_z_dL_to_m1_m2_source

# @jaxtyped
class NFPrior(Prior):
    
    nf: Transformed
    
    def __repr__(self):
        return f"NFPrior()"

    def __init__(
        self,
        nf_path: str,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs):
        super().__init__(naming, transforms)
        
        # Load the normalizing flow kwargs to construct like_flow
        logger.info("NFPrior -- Reading the NF kwargs")
        nf_kwargs_path = nf_path.replace(".eqx", "_kwargs.json")
        with open(nf_kwargs_path, "r") as f:
            nf_kwargs = json.load(f)
        print(f"The NF kwargs are: {nf_kwargs}")
            
        like_flow = block_neural_autoregressive_flow(
            key=jax.random.PRNGKey(0),
            base_dist=Normal(jnp.zeros(len(naming))),
            nn_depth=nf_kwargs["nn_depth"],
            nn_block_dim=nf_kwargs["nn_block_dim"],
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
        x_array = jnp.array([x[name] for name in self.naming]).T
        return self.nf.log_prob(x_array)
    
class DiracDeltaPrior(Prior):
    """
    A Dirac delta prior that returns a fixed value.
    """
    
    value: Float

    def __init__(self, 
                 value: Float,
                 naming: list[str],
                 transforms: dict[str, tuple[str, Callable]] = {},
                 **kwargs):
        self.value = value
        super().__init__(naming, transforms)

    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> dict[str, Float[Array, " n_samples"]]:
        samples = self.value * jnp.ones(n_samples)
        return self.add_name(samples[None])
    
    def log_prob(self, x: dict[str, Array]) -> Float:
        return 0.0
    

# @jaxtyped
class MyConditionalPrior(Prior):
    """This is specific for Mc_det, q to Lambda_1, Lambda_2 -- not designed for general use"""
    
    nf: Transformed
    base_prior: Composite
    
    def __repr__(self):
        return f"NFConditionalPrior()"

    def __init__(
        self,
        Mc_bounds: tuple[float, float],
        nf_path: str,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs):
        
        super().__init__(naming, transforms)
        
        # Initialize the base priors
        Mc_prior = Uniform(Mc_bounds[0], Mc_bounds[1], naming=["M_c"])
        
        # FIXME: hardcoded for now, generalize later
        q_prior = Uniform(0.125, 1.0, naming=["q"])
        d_L_prior = PowerLaw(1.0, 600.0, 2.0, naming=["d_L"])
        H0_prior = Uniform(40.0, 140.0, naming=["H0"])
        self.base_prior = Composite([Mc_prior, q_prior, d_L_prior, H0_prior])

        # Load the normalizing flow kwargs to construct like_flow
        logger.info("NFConditionalPrior -- Reading the NF kwargs")
        nf_kwargs_path = nf_path.replace(".eqx", "_kwargs.json")
        with open(nf_kwargs_path, "r") as f:
            nf_kwargs = json.load(f)
        print(f"The NF kwargs are: {nf_kwargs}")
            
        like_flow = block_neural_autoregressive_flow(
            key=jax.random.PRNGKey(0),
            base_dist=Normal(jnp.zeros(2)),
            cond_dim=2,
            nn_depth=nf_kwargs["nn_depth"],
            nn_block_dim=nf_kwargs["nn_block_dim"],
        )
        
        # Load the normalizing flow
        logger.info("Initializing the NF prior: deserializing leaves")
        _nf: Transformed = eqx.tree_deserialise_leaves(nf_path, like=like_flow)
        logger.info("Initializing the NF prior: deserializing leaves DONE")
        self.nf = _nf

    def sample_nf(self, rng_key: PRNGKeyArray, base_samples: dict[str, Float[Array, " n_samples"]]):
        """Sample the NF given the base samples from which we compute source frame masses on which the NF is then conditioned."""
        # Convert to source-frame masses
        m_1_source, m_2_source = Mc_q_H0_dL_to_m1_m2_source(base_samples)
        
        # Use the seed to sample the NF
        condition_samples = jnp.array([m_1_source, m_2_source]).T
        nf_samples = self.nf.sample(rng_key, (), condition=condition_samples)
        nf_samples = nf_samples.T
        
        _lambda_1, _lambda_2 = nf_samples[0], nf_samples[1]
        
        # Ensure lambda2 > lambda1
        lambda_1 = jnp.minimum(_lambda_1, _lambda_2)
        lambda_2 = jnp.maximum(_lambda_1, _lambda_2)
        
        # Clip to avoid negative values
        lambda_1 = jnp.clip(lambda_1, 0.1)
        lambda_2 = jnp.clip(lambda_2, 0.1)
        
        return_samples = {}
        return_samples["lambda_1"] = lambda_1
        return_samples["lambda_2"] = lambda_2
        
        return return_samples

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
        
        # Sample the masses:
        samples = self.base_prior.sample(rng_key, n_samples)
        nf_samples = self.sample_nf(rng_key, samples)
        
        samples["lambda_1"] = nf_samples["lambda_1"]
        samples["lambda_2"] = nf_samples["lambda_2"]
        
        return samples

    def log_prob(self, x: dict[str, Array]) -> Float:
        base_log_prob = self.base_prior.log_prob(x)
        
        m_1_source, m_2_source = Mc_q_H0_dL_to_m1_m2_source(x)
        x_array = jnp.array([x["lambda_1"], x["lambda_2"]]).T
        u_array = jnp.array([m_1_source, m_2_source]).T
        nf_log_prob = self.nf.log_prob(x_array, condition=u_array)
        return base_log_prob + nf_log_prob