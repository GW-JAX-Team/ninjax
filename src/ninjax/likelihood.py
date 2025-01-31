"""Gather all likelihoods that can be used by a ninjax program"""

import copy
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable

from jimgw.base import LikelihoodBase
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD

from ninjax.pipes.pipe_utils import logger

class ZeroLikelihood(LikelihoodBase):
    """Empty likelihood that constantly returns 0.0"""
    def __init__(self):
        super().__init__()
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0

class LikelihoodWithTransforms(LikelihoodBase):
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