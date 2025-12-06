"""Gather all likelihoods that can be used by a ninjax program

TODO: CRITICAL - This file needs major cleanup for new jim API!

      DEPRECATED classes:
      - LikelihoodWithTransforms: NO LONGER USED in new jim API
        Transforms are now passed directly to Jim constructor as separate pipelines

      POTENTIALLY USEFUL classes:
      - ZeroLikelihood: Could be useful for testing/debugging
      - CombinedLikelihood: Commented out, might be useful for multi-event analysis

      ACTION REQUIRED:
      1. Remove LikelihoodWithTransforms class (deprecated)
      2. Test if ZeroLikelihood is used anywhere
      3. Consider implementing CombinedLikelihood if needed for multi-event
      4. Add new likelihood classes here if custom likelihoods are needed
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable

from jimgw.core.base import LikelihoodBase
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD, BaseTransientLikelihoodFD

class ZeroLikelihood(LikelihoodBase):
    """Empty likelihood that constantly returns 0.0

    TODO: TEST - Verify if this is used anywhere in the codebase
          If not used, consider removing to reduce code complexity
    """
    def __init__(self):
        super().__init__()

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0

class LikelihoodWithTransforms(LikelihoodBase):
    """DEPRECATED: This class is NO LONGER USED in the new jim API!

    TODO: REMOVE - This entire class should be deleted.
          In the new API, transforms are passed directly to Jim constructor:
          - sample_transforms: list for MCMC efficiency
          - likelihood_transforms: list for parameter conversion
    """
    """Call an original likelihood but with some transforms applied to the parameters before evaluate"""
    def __init__(self, 
                 likelihood: LikelihoodBase, 
                 transforms: list[Callable],
                 temperature_schedule: Callable = None):
        self.likelihood = likelihood
        self.transforms = transforms
        self.required_keys = likelihood.required_keys
        
        if temperature_schedule is None:
            temperature_schedule = lambda x: 1.0
            
        self.temperature_schedule = temperature_schedule
        
    def transform(self, params: dict[str, Float]) -> dict[str, Float]:
        for transform in self.transforms:
            params = transform(params)
        return params
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # We make a safe copy of the params to avoid modifying the original
        inner_params = {}
        inner_params.update(params)
        inner_params = self.transform(inner_params)
        return self.likelihood.evaluate(inner_params, data)
    
    
# FIXME: might be removed!
# class CombinedLikelihood(LikelihoodBase):
#     """Likelihood class that combines multiple likelihoods into one and evaluates them all. Its log likelihood is the sum of the log likelihoods of the individual likelihoods."""
    
#     def __init__(self,
#                  likelihoods_list: list[LikelihoodBase]):
#         super().__init__()
#         self.likelihoods_list = likelihoods_list
        
#     def evaluate(self, params: dict[str, Float], data: dict) -> Float:
#         all_log_likelihoods = jnp.array([likelihood.evaluate(params, data) for likelihood in self.likelihoods_list])
#         return jnp.sum(all_log_likelihoods)
    