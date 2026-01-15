"""Main ninjax pipeline - orchestrates config, prior, transforms, likelihood, and sampler setup

TODO: CRITICAL AREAS FOR IMPROVEMENT:

      TESTED FEATURES (example_1):
      - Config loading and merging with defaults
      - Prior setup from .prior files
      - Transform pipeline setup (sample_transforms, likelihood_transforms)
      - HeterodynedTransientLikelihoodFD likelihood setup
      - flowMC hyperparameter configuration
      - Chirp mass prior recentering

      UNTESTED/INCOMPLETE FEATURES:
      - BaseTransientLikelihoodFD (non-heterodyned) likelihood setup
      - DoubleTransientLikelihoodFD and HeterodynedDoubleTransientLikelihoodFD (overlapping)
      - NF model kwargs handling (currently returns empty dict)
      - Non-GW likelihoods (if any are needed)
      - Validation of prior-transform-likelihood consistency
      - Sample transforms (currently empty list)

      POTENTIAL IMPROVEMENTS:
      1. Automatic waveform parameter extraction from likelihood.required_keys (removed in new API)
      2. Better validation of config file completeness
      3. Modular transform setup (config-driven instead of hardcoded)
      4. Automated tests for different likelihood configurations
      5. Refactor set_original_likelihood to reduce code duplication
"""

import os
import json
import numpy as np
from astropy.time import Time
import time

from jimgw.core.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD
from jimgw.core.jim import Jim
from jimgw.core.single_event.detector import Detector, GroundBased2G
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    BaseTransientLikelihoodFD,
    PhaseMarginalizedLikelihoodFD,
    HeterodynedPhaseMarginalizedLikelihoodFD,
)
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.prior import *
from jimgw.core.base import LikelihoodBase

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger
from ninjax.pipes.gw_pipe import GWPipe
from ninjax.parser import ConfigParser
from ninjax.prior import *
# NOTE: LikelihoodWithTransforms and transforms module no longer used in new API

import optax

# TODO: can we make this more automated?
LIKELIHOODS_DICT = {
    "BaseTransientLikelihoodFD": BaseTransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "PhaseMarginalizedLikelihoodFD": PhaseMarginalizedLikelihoodFD,
    "HeterodynedPhaseMarginalizedLikelihoodFD": HeterodynedPhaseMarginalizedLikelihoodFD,
}
GW_LIKELIHOODS = [
    "BaseTransientLikelihoodFD",
    "HeterodynedTransientLikelihoodFD",
    "PhaseMarginalizedLikelihoodFD",
    "HeterodynedPhaseMarginalizedLikelihoodFD",
]

# Deprecated keys from old jim API that should raise errors
DEPRECATED_KEYS = {
    "n_loop_training": "n_training_loops",
    "n_loop_production": "n_production_loops",
    "max_samples": "n_max_examples",
    "train_thinning": "local_thinning",
    "output_thinning": "global_thinning",
    "eps_mass_matrix": "mala_step_size",
    "num_layers": "rq_spline_n_layers",
    "hidden_size": "rq_spline_hidden_units",
    "num_bins": "rq_spline_n_bins",
    # Removed parameters (no replacement)
    "momentum": None,
    "use_global": None,
    "keep_quantile": None,
    "n_sample_max": None,
    "local_sampler_arg": None,
    "nf_model_kwargs": None,
}


class NinjaxPipe(object):
    
    def __init__(self, outdir: str):
        """Loads the config file and sets up the JimPipe object."""
        
        # Check if the output directory is valid
        logger.info("Checking and setting outdir")
        if not self.check_valid_outdir(outdir):
            raise ValueError(f"Outdir {outdir} must exist and must contain 'config.ini' and 'prior.prior'")
        self.outdir = outdir
        
        logger.info("Loading the given config")
        self.config = self.load_config()
        self.config["outdir"] = self.outdir
        self.dump_complete_config()
        
        # Setting some of the hyperparameters and the setup
        self.seed = self.get_seed()
        self.sampling_seed = self.get_sampling_seed()
        self.run_sampler = eval(self.config["run_sampler"])
        self.jim_hyperparameters, self.analysis_config = self.set_flowmc_hyperparameters()
        
        logger.info("Loading the priors")
        self.complete_prior = self.set_prior()
        self.naming = self.complete_prior.parameter_names
        self.n_dim = len(self.naming)
        # Prior bounds are no longer used in the new jim API
        logger.info("Finished prior setup")
        
        # Set the transforms using new jim API
        logger.info(f"Setting the transforms")
        self.sample_transforms = self.set_sample_transforms()
        self.likelihood_transforms = self.set_likelihood_transforms()

        # Finally, create the likelihood
        logger.info(f"Setting the likelihood")
        likelihood_str: str = self.config["likelihood"]
        self.check_valid_likelihood(likelihood_str)
        self.likelihood = self.set_original_likelihood(likelihood_str)

        # NOTE: Transforms are now passed directly to Jim constructor in the new API
        # The LikelihoodWithTransforms wrapper has been removed

        # Recenter chirp mass prior if enabled (must be done after injection is generated)
        if likelihood_str in GW_LIKELIHOODS and hasattr(self, 'gw_pipe') and self.gw_pipe.is_gw_injection:
            logger.info("Checking if chirp mass prior recentering is enabled")
            self.recenter_chirp_mass_prior(self.gw_pipe.gw_injection)

        # TODO: check if the setup prior -> transform -> likelihood is OK
        # TODO: required_keys attribute removed in new jim API - skipping this check for now
        # logger.info(f"Required keys for the likelihood: {self.likelihood.required_keys}")
        # self.check_prior_transforms_likelihood_setup()

        # TODO: make the default keys to plot empty/None and use prior naming in that case
        logger.info(f"Will plot these keys: {self.keys_to_plot}")
        self.labels_to_plot = []
        recognized_labels = list(utils.LABELS_TRANSLATION_DICT.keys())
        for key in self.keys_to_plot:
            if key in recognized_labels:
                self.labels_to_plot.append(utils.LABELS_TRANSLATION_DICT[key])
            else:
                logger.info(f"Plot key {key} does not have a known LaTeX translation")
                self.labels_to_plot.append(key)
        logger.info(f"Will plot the labels: {self.labels_to_plot}")
        logger.info(f"Ninjax setup complete.")
        
    
    @property
    def outdir(self):
        return self._outdir
    
    @property
    def keys_to_plot(self):
        keys = self.config["keys_to_plot"]
        keys = keys.split(",")
        keys = [k.strip() for k in keys]
        return keys
    
    def check_valid_outdir(self, outdir: str) -> bool:
        """Check if the outdir exists and contains required files."""
        if not os.path.isdir(outdir):
            return False
        self.config_filename = os.path.join(outdir, "config.ini")
        self.prior_filename = os.path.join(outdir, "prior.prior")
        return all([os.path.isfile(self.config_filename), os.path.isfile(self.prior_filename)])
    
    @outdir.setter
    def outdir(self, outdir: str):
        logger.info(f"The outdir is set to {outdir}")
        self._outdir = outdir
        
    @property
    def nf_model_kwargs(self) -> dict:
        kwargs = eval(self.config["nf_model_kwargs"])
        if kwargs is None:
            return {}
        logger.info(f"Setting the NF model kwargs to {kwargs}")
        return kwargs
        
    def load_config(self) -> dict:
        """Set the configuration by parsing the user and default config files."""

        parser = ConfigParser()
        config_filename = os.path.join(self.outdir, "config.ini")
        user_config: dict = parser.parse(config_filename)

        # Parse the default config for non-specified keys
        default_config_filename = os.path.join(os.path.dirname(__file__), "default_config.ini")
        config: dict = parser.parse(default_config_filename)

        # Check for deprecated keys and throw errors with helpful messages
        deprecated_keys_found = set(user_config.keys()) & set(DEPRECATED_KEYS.keys())
        if len(deprecated_keys_found) > 0:
            error_messages = []
            for key in deprecated_keys_found:
                replacement = DEPRECATED_KEYS[key]
                if replacement is not None:
                    error_messages.append(f"  - '{key}' is deprecated, use '{replacement}' instead")
                else:
                    error_messages.append(f"  - '{key}' is deprecated and no longer supported (remove it)")

            raise ValueError(
                f"Deprecated configuration keys found in {config_filename}:\n" +
                "\n".join(error_messages) +
                "\n\nPlease update your config.ini file to use the new jim API parameter names."
            )

        recognized_keys = set(config.keys())
        unrecognized_keys = set(user_config.keys()) - recognized_keys
        if len(unrecognized_keys) > 0:
            logger.warn(f"Unrecognized keys given: {unrecognized_keys}. These will be ignored")

        # Drop the unrecognized keys
        for key in unrecognized_keys:
            user_config.pop(key)

        config.update(user_config)
        logger.info(f"Arguments loaded into the config: {config}")

        return config
        
    def dump_complete_config(self):
        """Dumps the complete config after merging the user and the default settings to a JSON file"""
        complete_ini_filename = os.path.join(self.outdir, "complete_config.json")
        json.dump(self.config, open(complete_ini_filename, "w"), indent=4, cls=utils.CustomJSONEncoder)
        logger.info(f"Complete config file written to {os.path.abspath(complete_ini_filename)}")

    def set_prior(self) -> CombinePrior:
        prior_list = []
        with open(self.prior_filename, "r") as f:
            for line in f:
                stripped_line = line.strip()

                if stripped_line == "":
                    logger.info("Encountered empty line in prior file, continue")
                    continue

                # Skip lines that are commented out
                if stripped_line.startswith("#"):
                    continue

                logger.info(f"   {stripped_line}")
                exec(stripped_line)

                prior_name = stripped_line.split("=")[0].strip()
                prior_list.append(eval(prior_name))

        # Store the prior list for potential recentering later
        self.prior_list = prior_list
        return CombinePrior(prior_list)

    def recenter_chirp_mass_prior(self, injected_params: dict):
        """
        Recenter the chirp mass prior around the injected value.
        This creates a narrow prior for sampling while keeping the wide prior for injection.

        Args:
            injected_params: Dictionary containing the injected parameters
        """
        if not eval(self.config.get("center_chirp_mass_prior", "False")):
            logger.info("Chirp mass prior recentering disabled")
            return

        if 'M_c' not in injected_params:
            logger.warning("M_c not found in injected parameters, skipping chirp mass prior recentering")
            return

        injected_M_c = injected_params['M_c']
        delta = float(self.config["chirp_mass_prior_delta"])

        # Calculate new bounds
        M_c_min = injected_M_c - delta
        M_c_max = injected_M_c + delta

        logger.info(f"Recentering chirp mass prior around injected value: M_c = {injected_M_c}")
        logger.info(f"New chirp mass prior bounds: [{M_c_min}, {M_c_max}] (delta = {delta})")

        # Find and replace the M_c prior in the prior list
        new_prior_list = []
        M_c_prior_found = False

        for prior in self.prior_list:
            # Check if this prior is for M_c
            if hasattr(prior, 'parameter_names') and 'M_c' in prior.parameter_names:
                # Create a new centered prior for M_c
                new_M_c_prior = UniformPrior(M_c_min, M_c_max, parameter_names=['M_c'])
                new_prior_list.append(new_M_c_prior)
                M_c_prior_found = True
                logger.info(f"Replaced M_c prior with centered prior: UniformPrior({M_c_min}, {M_c_max})")
            else:
                new_prior_list.append(prior)

        if not M_c_prior_found:
            logger.warning("M_c prior not found in prior list, skipping recentering")
            return

        # Rebuild the complete prior with the new M_c prior
        self.prior_list = new_prior_list
        self.complete_prior = CombinePrior(new_prior_list)
        logger.info("Successfully recentered chirp mass prior")

    def set_prior_bounds(self):
        # TODO: generalize this: (i) only for GW relative binning, (ii) xmin xmax might fail for more advanced priors
        return jnp.array([[p.xmin, p.xmax] for p in self.complete_prior.priors])
    
    
    def set_flowmc_hyperparameters(self) -> tuple[dict, dict]:
        """
        Returns two dicts:
        1. jim_hyperparameters - parameters passed to Jim constructor
        2. analysis_config - parameters used elsewhere (analysis.py, plotting, etc.)
        """
        # Parameters passed to Jim constructor (updated for new Jim API)
        jim_hyperparameters = {
            # Core sampling parameters
            "n_chains": int(self.config["n_chains"]),
            "n_local_steps": int(self.config["n_local_steps"]),
            "n_global_steps": int(self.config["n_global_steps"]),
            "n_training_loops": int(self.config["n_training_loops"]),
            "n_production_loops": int(self.config["n_production_loops"]),
            "n_epochs": int(self.config["n_epochs"]),

            # MALA step size (will be processed below for mala_step_size_scale)
            "mala_step_size": float(self.config["mala_step_size"]),

            # Batch parameters
            "chain_batch_size": int(self.config["chain_batch_size"]),
            "batch_size": int(self.config["batch_size"]),
            "n_max_examples": int(self.config["n_max_examples"]),

            # Normalizing flow parameters
            "rq_spline_hidden_units": [int(x) for x in self.config["rq_spline_hidden_units"].split(",")],
            "rq_spline_n_bins": int(self.config["rq_spline_n_bins"]),
            "rq_spline_n_layers": int(self.config["rq_spline_n_layers"]),
            "learning_rate": float(self.config["learning_rate"]),

            # Thinning parameters
            "local_thinning": int(self.config["local_thinning"]),
            "global_thinning": int(self.config["global_thinning"]),

            # Other sampling parameters
            "n_NFproposal_batch_size": int(self.config["n_NFproposal_batch_size"]),
            "history_window": int(self.config["history_window"]),

            # Parallel tempering parameters
            "n_temperatures": int(self.config["n_temperatures"]),
            "max_temperature": float(self.config["max_temperature"]),
            "n_tempered_steps": int(self.config["n_tempered_steps"]),

            # Verbose flag
            "verbose": eval(self.config["verbose"]),
        }

        # Handle mala_step_size_scale if provided
        mala_step_size_scale = eval(self.config["mala_step_size_scale"])
        if mala_step_size_scale is not None:
            logger.info(f"Applying mala_step_size_scale: {mala_step_size_scale}")
            # Create a per-parameter step size array
            # This will be implemented after prior is set up (deferred to analysis.py)
            jim_hyperparameters["mala_step_size_scale"] = mala_step_size_scale

        # Parameters NOT passed to Jim (used elsewhere in analysis.py)
        analysis_config = {
            "save_training_chains": eval(self.config["save_training_chains"]),
            "use_scheduler": eval(self.config["use_scheduler"]),
        }

        return jim_hyperparameters, analysis_config
    
    def set_sample_transforms(self) -> list:
        """
        Build the sample_transforms pipeline for the new jim API.
        These transforms operate on the sampling space for MCMC efficiency.

        For now, we return an empty list as we don't need sample transforms
        for the basic example_1 test case. In the future, we could add:
        - PeriodicTransform for periodic parameters (phase_c, ra, psi)
        - Other transforms for improving MCMC sampling efficiency

        TODO: IMPROVEMENT - Make this configurable from config file!
              Currently hardcoded to return empty list
              Should support:
              1. PeriodicTransform for periodic params (from jim.transforms)
              2. BoundToUnbound transforms for bounded parameters
              3. Config-driven transform specification
        """
        sample_transforms = []
        logger.info(f"Built sample_transforms pipeline with {len(sample_transforms)} transforms")
        return sample_transforms

    def set_likelihood_transforms(self) -> list:
        """
        Build the likelihood_transforms pipeline for the new jim API.
        These transforms convert from prior space to likelihood space.

        For example_1 (BNS with aligned spins), we need:
        - q → eta (mass ratio to symmetric mass ratio) for waveform evaluation

        Note: cos_iota → iota and sin_dec → dec transforms are NO LONGER NEEDED
        because we now use SinePrior and CosinePrior which sample directly in the correct space!

        TODO: CRITICAL - HARDCODED! This assumes all runs need MassRatioToSymmetricMassRatioTransform
              Should be:
              1. Inferred from waveform model requirements
              2. Specified in config file
              3. Automatically determined from prior parameter names
              4. Support for other transforms (spin transforms, mass transforms, etc.)
        """
        likelihood_transforms = [
            MassRatioToSymmetricMassRatioTransform,  # q → eta
        ]
        logger.info(f"Built likelihood_transforms pipeline with {len(likelihood_transforms)} transforms")
        return likelihood_transforms
    
    @staticmethod
    def check_valid_likelihood(likelihood_str) -> None:
        if likelihood_str not in LIKELIHOODS_DICT:
            raise ValueError(f"Likelihood {likelihood_str} not supported. Supported likelihoods are {list(LIKELIHOODS_DICT.keys())}.")

    def set_original_likelihood(self, likelihood_str: str) -> LikelihoodBase:
        """Create the likelihood object depending on the given likelihood string."""
        
        # Set up everything needed for GW likelihood
        if likelihood_str in GW_LIKELIHOODS:
            logger.info("GW likelihood provided, setting up the GW pipe")
            # TODO: this is becoming quite cumbersome... perhaps there is a better way to achieve this?
            self.config["gw_is_overlapping"] = likelihood_str in ["DoubleTransientLikelihoodFD", "HeterodynedDoubleTransientLikelihoodFD"]
            self.gw_pipe = GWPipe(self.config, self.outdir, self.complete_prior, self.seed, self.likelihood_transforms)
            
        # Create the likelihood
        if likelihood_str == "HeterodynedTransientLikelihoodFD":
            logger.info("Using GW HeterodynedTransientLikelihoodFD. Initializing likelihood")
            
            # Check what to do in case of an injection
            if self.gw_pipe.is_gw_injection:
                if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                    ref_params = self.gw_pipe.gw_injection
                    logger.info("Using the injection parameters as reference parameters for the relative binning")
                else:
                    ref_params = None
                    logger.info("Will search for reference waveform for relative binning")
                    
            # Check what to do in case of analyzing real data
            else:
                if self.gw_pipe.relative_binning_ref_params is not None or self.gw_pipe.relative_binning_ref_params != "None":
                    ref_params = self.gw_pipe.relative_binning_ref_params
                    logger.info("Using provided reference parameters for relative binning")
                else:
                    ref_params = None
                    logger.info("Will search for reference waveform for relative binning")
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")

            # Apply likelihood transforms to ref_params if needed
            if ref_params is not None and self.likelihood_transforms:
                logger.info("Applying likelihood transforms to reference parameters")
                transformed_ref_params = ref_params.copy()
                for transform in self.likelihood_transforms:
                    transformed_ref_params = transform.forward(transformed_ref_params)
                ref_params = transformed_ref_params
                logger.info(f"Transformed ref_params keys: {list(ref_params.keys())}")

            init_heterodyned_start = time.time()
            likelihood = HeterodynedTransientLikelihoodFD(
                self.gw_pipe.ifos,
                self.gw_pipe.waveform,
                f_min=self.gw_pipe.fmin,
                f_max=self.gw_pipe.fmax,
                trigger_time=self.gw_pipe.trigger_time,
                n_bins=self.gw_pipe.relative_binning_binsize,
                ref_params=ref_params,
                reference_waveform=self.gw_pipe.reference_waveform,
                prior=self.complete_prior,
                **self.gw_pipe.kwargs
                )
            init_heterodyned_end = time.time()
            
            logger.info(f"Initialization of HeterodynedTransientLikelihoodFD took {init_heterodyned_end - init_heterodyned_start} seconds = {(init_heterodyned_end - init_heterodyned_start) / 60} minutes")

            # TODO: required_keys removed in new API
            # print(likelihood.required_keys)

        elif likelihood_str == "PhaseMarginalizedLikelihoodFD":
            logger.info("Using GW PhaseMarginalizedLikelihoodFD (non-heterodyned, phase-marginalized). Initializing likelihood")

            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")

            likelihood = PhaseMarginalizedLikelihoodFD(
                self.gw_pipe.ifos,
                self.gw_pipe.waveform,
                f_min=self.gw_pipe.fmin,
                f_max=self.gw_pipe.fmax,
                trigger_time=self.gw_pipe.trigger_time,
                **self.gw_pipe.kwargs
            )
            logger.info("PhaseMarginalizedLikelihoodFD initialized successfully")

        elif likelihood_str == "HeterodynedPhaseMarginalizedLikelihoodFD":
            logger.info("Using GW HeterodynedPhaseMarginalizedLikelihoodFD (heterodyned + phase-marginalized). Initializing likelihood")

            # Check what to do in case of an injection
            if self.gw_pipe.is_gw_injection:
                if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                    ref_params = self.gw_pipe.gw_injection
                    logger.info("Using the injection parameters as reference parameters for the relative binning")
                else:
                    ref_params = None
                    logger.info("Will search for reference waveform for relative binning")

            # Check what to do in case of analyzing real data
            else:
                if self.gw_pipe.relative_binning_ref_params is not None or self.gw_pipe.relative_binning_ref_params != "None":
                    ref_params = self.gw_pipe.relative_binning_ref_params
                    logger.info("Using provided reference parameters for relative binning")
                else:
                    ref_params = None
                    logger.info("Will search for reference waveform for relative binning")

            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")

            # Apply likelihood transforms to ref_params if needed
            if ref_params is not None and self.likelihood_transforms:
                logger.info("Applying likelihood transforms to reference parameters")
                transformed_ref_params = ref_params.copy()
                for transform in self.likelihood_transforms:
                    transformed_ref_params = transform.forward(transformed_ref_params)
                ref_params = transformed_ref_params
                logger.info(f"Transformed ref_params keys: {list(ref_params.keys())}")

            init_heterodyned_start = time.time()
            likelihood = HeterodynedPhaseMarginalizedLikelihoodFD(
                self.gw_pipe.ifos,
                self.gw_pipe.waveform,
                f_min=self.gw_pipe.fmin,
                f_max=self.gw_pipe.fmax,
                trigger_time=self.gw_pipe.trigger_time,
                n_bins=self.gw_pipe.relative_binning_binsize,
                ref_params=ref_params,
                reference_waveform=self.gw_pipe.reference_waveform,
                prior=self.complete_prior,
                **self.gw_pipe.kwargs
            )
            init_heterodyned_end = time.time()

            logger.info(f"Initialization of HeterodynedPhaseMarginalizedLikelihoodFD took {init_heterodyned_end - init_heterodyned_start} seconds = {(init_heterodyned_end - init_heterodyned_start) / 60} minutes")

        elif likelihood_str == "TransientLikelihoodFD":
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            likelihood = TransientLikelihoodFD(
                self.gw_pipe.ifos,
                waveform=self.gw_pipe.waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                **self.gw_pipe.kwargs
                )
            # TODO: required_keys removed in new API
            # print(likelihood.required_keys)
        
        elif likelihood_str == "DoubleTransientLikelihoodFD":
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            likelihood = DoubleTransientLikelihoodFD(
                self.gw_pipe.ifos,
                waveform=self.gw_pipe.waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                **self.gw_pipe.kwargs
                )
            # TODO: required_keys removed in new API
            # print(likelihood.required_keys)
        
        elif likelihood_str == "HeterodynedDoubleTransientLikelihoodFD":
            if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                ref_params = self.gw_pipe.gw_injection
                logger.info("Using the true parameters as reference parameters for the relative binning")
            else:
                ref_params = None
                logger.info("Will search for reference waveform for relative binning")
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            init_heterodyned_start = time.time()
            likelihood = HeterodynedDoubleTransientLikelihoodFD(
                self.gw_pipe.ifos,
                self.gw_pipe.waveform,
                f_min=self.gw_pipe.fmin,
                f_max=self.gw_pipe.fmax,
                trigger_time=self.gw_pipe.trigger_time,
                n_bins=self.gw_pipe.relative_binning_binsize,
                ref_params=ref_params,
                reference_waveform=self.gw_pipe.reference_waveform,
                prior=self.complete_prior,
                **self.gw_pipe.kwargs
                )
            # TODO: required_keys removed in new API
            # print(likelihood.required_keys)
            init_heterodyned_end = time.time()
            
            logger.info(f"Initialization of HeterodynedTransientLikelihoodFD took around {int((init_heterodyned_end - init_heterodyned_start) / 60)} minutes")
        
        return likelihood
    
    def check_prior_transforms_likelihood_setup(self):
        """Check if the setup between prior, transforms, and likelihood is correct by a small test."""
        logger.info("Checking the setup between prior, transforms, and likelihood")
        sample = self.complete_prior.sample(jax.random.PRNGKey(self.seed), 3)
        logger.info(f"sample: {sample}")
        # sample_transformed = jax.vmap(self.likelihood.transform)(sample)
        # logger.info(f"sample_transformed: {sample_transformed}")
        
        # TODO: what if we actually need to give data instead of nothing?
        log_prob = jax.vmap(self.likelihood.evaluate)(sample, {})
        if jnp.isnan(log_prob).any():
            raise ValueError("Log probability is NaN. Something is wrong with the setup!")
        logger.info(f"log_prob: {log_prob}")

    def get_seed(self):
        if isinstance(self.config["seed"], int):
            return self.config["seed"]
        seed = eval(self.config["seed"])
        if seed is None:
            seed = np.random.randint(0, 999999)
            logger.info(f"No seed specified. Generating a random seed: {seed}")
        self.config["seed"] = seed
        return seed

    def get_sampling_seed(self):
        if isinstance(self.config["sampling_seed"], int):
            return self.config["sampling_seed"]
        sampling_seed = eval(self.config["sampling_seed"])
        if sampling_seed is None:
            sampling_seed = np.random.randint(0, 999999)
            logger.info(f"No sampling_seed specified. Generating a random sampling_seed: {sampling_seed}")
        self.config["sampling_seed"] = sampling_seed
        return sampling_seed