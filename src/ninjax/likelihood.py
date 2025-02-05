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