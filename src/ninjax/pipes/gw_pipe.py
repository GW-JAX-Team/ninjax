"""GW-specific pipeline setup for ninjax

TODO: CRITICAL ISSUES - This file has several untested features and hardcoded assumptions

      TESTED FEATURES (example_1):
      - Basic injection-recovery with H1, L1, V1
      - TaylorF2 and IMRPhenomD_NRTidalv2 waveforms
      - HeterodynedTransientLikelihoodFD likelihood
      - SNR threshold filtering for injections
      - PSD loading from files
      - Chirp mass prior recentering

      UNTESTED FEATURES (may be broken):
      - ET and CE detectors (now available but untested)
      - BaseTransientLikelihoodFD likelihood (non-heterodyned)
      - Overlapping signal injections (set_overlapping_gw_injection)
      - Real data loading (set_gw_data_from_npz)
      - IMRPhenomPv2 precessing waveform
      - Non-BNS waveforms (IMRPhenomD for BBH)
      - Duration auto-computation for non-BNS systems
      - EOS file injection for BNS systems
      - Relative binning with user-provided reference parameters

      REMOVED FEATURES (from old API):
      - TransientLikelihoodFD class name (use BaseTransientLikelihoodFD)
      - TriangularNetwork2G class (removed)

      DETECTOR SUPPORT:
      - H1, L1, V1: Fully tested ✅
      - ET: Available (returns 3 detectors: ET1, ET2, ET3 for triangular config) ⚠️ UNTESTED
      - CE: Available ⚠️ UNTESTED

      ACTION REQUIRED:
      1. Test ET and CE detector support
      2. Test all other untested features listed above
      3. Add validation for waveform model parameter requirements
      4. Improve error messages when detector/waveform combos are invalid
      5. Add automated tests for injection-recovery workflows
      6. Refactor to reduce code duplication between overlapping and single injections
"""

import os
import json
from typing import Callable
import numpy as np
from astropy.time import Time
import jax
import jax.numpy as jnp

from jimgw.core.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD, RippleIMRPhenomPv2, RippleTaylorF2QM_taper
from jimgw.core.single_event.detector import Detector, GroundBased2G, get_H1, get_L1, get_V1, get_ET, get_CE
from jimgw.core.prior import CombinePrior

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger

WAVEFORMS_DICT = {"TaylorF2": RippleTaylorF2, 
                  "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
                  "IMRPhenomD": RippleIMRPhenomD,
                  "IMRPhenomPv2": RippleIMRPhenomPv2,
                  "TaylorF2QM_taper": RippleTaylorF2QM_taper,  # Placeholder, replace with actual implementation
                  }

SUPPORTED_WAVEFORMS = list(WAVEFORMS_DICT.keys())
BNS_WAVEFORMS = ["IMRPhenomD_NRTidalv2", "TaylorF2"]

class GWPipe:
    def __init__(self,
                 config: dict,
                 outdir: str,
                 prior: CombinePrior,
                 seed: int,
                 likelihood_transforms: list[Callable]):
        self.config = config
        self.outdir = outdir
        self.complete_prior = prior
        self.seed = seed
        self.likelihood_transforms = likelihood_transforms
        
        # Initialize other GW-specific attributes
        self.eos_file = self.set_eos_file()
        self.is_BNS_run = self.waveform_approximant in BNS_WAVEFORMS
        self.psds_dict = self.set_psds_dict()
        self.ifos = self.set_ifos()
        self.waveform = self.set_waveform()
        self.reference_waveform = self.set_reference_waveform()
        
        # TODO: data loading if preprocesse data is shared
        # Check if an injection and if has to be loaded, or if provided GW data must be loaded
        self.is_gw_injection = eval(self.config["gw_injection"])
        logger.info(f"GW run is an injection")
        if self.is_gw_injection:
            if self.config["gw_is_overlapping"]:
                self.gw_injection = self.set_overlapping_gw_injection()
            else:
                # TODO: should separate load existing injection from creating new one
                self.gw_injection = self.set_gw_injection()
            self.dump_gw_injection()
            # TODO: this is not recommended for the ET BNS 5 Hz runs! Put a flag here to know when to dump or not to dump
            # self.dump_gw_data()
        else:
            self.set_gw_data_from_npz()
            # self.set_detector_info() # needed? Duration, epoch, gmst,...
            
        
            
    @property
    def fmin(self):
        return float(self.config["fmin"])

    @property
    def fmax(self):
        return float(self.config["fmax"])
    
    @property
    def fref(self):
        return float(self.config["fref"])
    
    @property
    def gw_load_existing_injection(self):
        return eval(self.config["gw_load_existing_injection"])

    @property
    def gw_SNR_threshold_low(self):
        return float(self.config["gw_SNR_threshold_low"])

    @property
    def gw_SNR_threshold_high(self):
        return float(self.config["gw_SNR_threshold_high"])

    @property
    def gw_max_injection_attempts(self):
        return int(self.config["gw_max_injection_attempts"])

    @property
    def post_trigger_duration(self):
        return float(self.config["post_trigger_duration"])
    
    @property
    def trigger_time(self):
        return float(self.config["trigger_time"])

    @property
    def waveform_approximant(self):
        return self.config["waveform_approximant"]
    
    @property
    def psd_file_H1(self):
        return self.config["psd_file_H1"]

    @property
    def psd_file_L1(self):
        return self.config["psd_file_L1"]

    @property
    def psd_file_V1(self):
        return self.config["psd_file_V1"]

    @property
    def psd_file_ET1(self):
        return self.config["psd_file_ET1"]
    
    @property
    def psd_file_ET2(self):
        return self.config["psd_file_ET2"]
    
    @property
    def psd_file_ET3(self):
        return self.config["psd_file_ET3"]
    
    @property
    def psd_file_CE(self):
        return self.config["psd_file_CE"]
    
    @property
    def data_file_H1(self):
        return self.config["data_file_H1"]
    
    @property
    def data_file_L1(self):
        return self.config["data_file_L1"]
    
    @property
    def data_file_V1(self):
        return self.config["data_file_V1"]
    
    @property
    def data_file_ET1(self):
        return self.config["data_file_ET1"]
    
    @property
    def data_file_ET2(self):
        return self.config["data_file_ET2"]
    
    @property
    def data_file_ET3(self):
        return self.config["data_file_ET3"]
    
    @property
    def relative_binning_binsize(self):
        return int(self.config["relative_binning_binsize"])
    
    @property
    def relative_binning_ref_params_equal_true_params(self):
        return eval(self.config["relative_binning_ref_params_equal_true_params"])
    
    @property
    def relative_binning_ref_params(self):
        return eval(self.config["relative_binning_ref_params"])
    
    @property
    def config_duration(self):
        return eval(self.config["duration"])
    
    @property
    def kwargs(self) -> dict:
        _kwargs = eval(self.config["gw_kwargs"])
        if _kwargs is None:
            return {}
        return _kwargs
    
    def set_psds_dict(self) -> dict:
        psds_dict = {"H1": self.psd_file_H1,
                     "L1": self.psd_file_L1,
                     "V1": self.psd_file_V1,
                     "ET1": self.psd_file_ET1,
                     "ET2": self.psd_file_ET2,
                     "ET3": self.psd_file_ET3,
                     "CE": self.psd_file_CE,
                     }
        return psds_dict
    
    def set_eos_file(self) -> str:
        """
        Check if an EOS file for the lambdas has been provided and if in correct format.
        Returns None if the provided file is not recognized.
        """
        
        # TODO: get full file path if needed
        eos_file = str(self.config["eos_file"])
        logger.info(f"eos_file is {eos_file}")
        if eos_file.lower() == "none" or len(eos_file) == 0:
            logger.info("No eos_file specified. Will sample lambdas uniformly.")
            return None
        else:
            self.check_valid_eos_file(eos_file)
            logger.info(f"Using eos_file {eos_file} for BNS injections")
            
        return eos_file
    
    def check_valid_eos_file(self, eos_file):
        """
        Check if the Lambdas EOS file has the right format, i.e. it should have "masses_EOS" and "Lambdas_EOS" keys.
        """
        if not os.path.exists(eos_file):
            raise ValueError(f"eos_file {eos_file} does not exist")
        if not eos_file.endswith(".npz"):
            raise ValueError("eos_file must be an npz file")
        data: dict = np.load(eos_file)
        keys = list(data.keys())
        if "masses_EOS" not in keys:
            raise ValueError("Key `masses_EOS` not found in eos_file")
        if "Lambdas_EOS" not in keys:
            raise ValueError("Key `Lambdas_EOS` not found in eos_file")
        
        return
        
    def set_gw_injection(self):
        """
        Function that creates a GW injection, taking into account the given priors and the SNR thresholds.
        If an existing injection.json exists, will load that one. 
        # TODO: do not hardcode injection.json, make more flexible

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Setting up GW injection . . . ")
        logger.info(f"The SNR thresholds are: {self.gw_SNR_threshold_low} - {self.gw_SNR_threshold_high}")
        logger.info(f"Maximum injection attempts allowed: {self.gw_max_injection_attempts}")
        pass_threshold = False
        attempt_counter = 0

        sample_key = jax.random.PRNGKey(self.seed)
        while not pass_threshold:
            
            # Generate the parameters or load them from an existing file
            injection_path = os.path.join(self.outdir, "injection.json")
            if self.gw_load_existing_injection:
                logger.info(f"Loading existing injection, path: {injection_path}")
                injection = json.load(open(injection_path))
                # When loading existing injection, it's already in untransformed (prior) space
                # The file was saved with 'q', 'M_c', etc. before transforms
                injection_for_plotting = injection.copy()
            else:
                logger.info(f"Generating new injection")
                sample_key, subkey = jax.random.split(sample_key)
                injection = utils.generate_injection(injection_path, self.complete_prior, subkey)
                # Save untransformed injection for plotting (in prior space with parameters like 'q')
                # This must be done BEFORE apply_transforms which converts q → eta
                injection_for_plotting = injection.copy()

            # TODO: here is where we might have to transform from prior to ripple/jim parameters

            # If a BNS run, we can infer Lambdas from a given EOS if desired and override the parameters
            if self.is_BNS_run and self.eos_file is not None:
                logger.info(f"Computing lambdas from EOS file {self.eos_file} . . . ")
                injection = utils.inject_lambdas_from_eos(injection, self.eos_file)
                injection_for_plotting = injection.copy()  # Update plotting version too

            # Get duration based on Mc and fmin if not specified
            if self.config_duration is None:
                # TODO: put a minimum of 4 seconds here in case of very short signals?
                duration = utils.signal_duration(self.fmin, injection["M_c"])
                duration = 2 ** np.ceil(np.log2(duration))
                duration = float(duration)
                logger.info(f"Duration is not specified in the config. Computed chirp time: for fmin = {self.fmin} and M_c = {injection['M_c']} is {duration}")
            else:
                duration = self.config_duration
                logger.info(f"Duration is specified in the config: {duration}")

            self.duration = duration

            # DON'T manually construct frequencies - let jim's Data objects handle this properly
            # Jim uses rfftfreq() which is numerically stable, while arange() has precision issues
            # self.frequencies will be set from detector data after inject_signal creates it

            # Make any necessary conversions
            # FIXME: hacky way for now --  if users specify iota in the injection, but sample over cos_iota and do the tfo, this breaks
            try:
                injection = self.apply_transforms(injection)
            except Exception as e:
                logger.error(f"Error in applying transforms: {e}")
                # raise ValueError("Error in applying transforms")
            
            logger.info("After transforms, the injection parameters are:")
            logger.info(injection)
            
            # Setup the timing setting for the injection
            self.epoch = self.duration - self.post_trigger_duration
            self.gmst = Time(self.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
            
            # Get the array of the injection parameters
            # TODO: required_keys attribute removed in new jim API - need to implement waveform parameter mapping
            # For now, using all injection parameters
            true_param = {key: float(injection[key]) for key in injection.keys()}

            # Add detector-specific parameters required by inject_signal
            true_param['gmst'] = self.gmst
            true_param['trigger_time'] = self.trigger_time

            logger.info(f"The trial injection parameters are {true_param}")

            self.detector_param = {
                'psi':    injection["psi"],
                't_c':    injection["t_c"],
                'ra':     injection["ra"],
                'dec':    injection["dec"],
                'epoch':  self.epoch,
                'gmst':   self.gmst,
                'trigger_time': self.trigger_time,
                }
            
            # Generating the geocenter waveform (not needed separately - inject_signal will call waveform)
            logger.info("Injecting signals . . .")

            # Calculate sampling frequency (Nyquist theorem)
            sampling_frequency = self.fmax * 2

            key = jax.random.PRNGKey(self.seed)
            logger.info("self.ifos")
            logger.info(self.ifos)
            for ifo in self.ifos:
                # Set frequency bounds BEFORE loading PSD or injecting signal
                # This is critical to avoid waveform evaluation at invalid frequencies (like f=0)
                ifo.frequency_bounds = (self.fmin, self.fmax)

                # Load PSD from file
                psd_file = self.psds_dict[ifo.name]

                # Check if PSD file is just a filename (not absolute path)
                # If so, look for it in ninjax/src/ninjax/ directory
                if not psd_file.startswith('/'):
                    # Get the directory where this file is located (ninjax/src/ninjax/pipes/)
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up one level to ninjax/src/ninjax/
                    parent_dir = os.path.dirname(current_dir)
                    psd_file = os.path.join(parent_dir, psd_file)

                logger.info(f"Loading PSD from file: {psd_file}")

                # Load PSD from file using jim's built-in method
                # This will load the PSD with its original frequencies
                # Jim will automatically interpolate it to match the Data frequencies
                ifo.load_and_set_psd(psd_file=psd_file)

                key, subkey = jax.random.split(key)
                # inject_signal will create proper Data with rfftfreq-generated frequencies
                # and jim will automatically interpolate the PSD to match
                ifo.inject_signal(
                    duration=self.duration,
                    sampling_frequency=sampling_frequency,
                    epoch=self.epoch,
                    waveform_model=self.waveform,
                    parameters=true_param,
                    is_zero_noise=False,
                    rng_key=subkey,
                )
                
                # TODO: remove once tested
                logger.info(f"Signal injected in ifo {ifo.name}. Frequencies, data, and PSD:")
                logger.info(ifo.frequencies)
                logger.info(ifo.data)
                logger.info(ifo.psd)

            # Set self.frequencies from the first detector's data (now properly initialized)
            # This is needed for compatibility with downstream code
            self.frequencies = self.ifos[0].frequencies

            # Generate the sky-frame waveform for SNR computation
            # Use the first detector's sliced frequencies (respecting fmin/fmax bounds)
            self.h_sky = self.waveform(self.ifos[0].sliced_frequencies, true_param)

            # Compute the SNRs, and save to a dict to be dumped later on
            snr_dict = {}
            for ifo in self.ifos:
                # if ifo.name == "ET":
                #     snr_dict["ET1_SNR"] = utils.compute_snr(ifo[0], self.h_sky, self.detector_param)
                #     snr_dict["ET2_SNR"] = utils.compute_snr(ifo[1], self.h_sky, self.detector_param)
                #     snr_dict["ET3_SNR"] = utils.compute_snr(ifo[2], self.h_sky, self.detector_param)
                # else:
                snr = utils.compute_snr(ifo, self.h_sky, self.detector_param)
                logger.info(f"SNR for ifo {ifo.name} is {snr}")
                snr_dict[f"{ifo.name}_SNR"] = snr
            
            snr_list = list(snr_dict.values())
            self.network_snr = float(jnp.sqrt(jnp.sum(jnp.array(snr_list) ** 2)))
            
            logger.info(f"The network SNR is {self.network_snr}")

            # If the SNR is too low, we need to generate new parameters
            pass_threshold = self.network_snr > self.gw_SNR_threshold_low and self.network_snr < self.gw_SNR_threshold_high
            if not pass_threshold:
                attempt_counter += 1
                logger.info(f"Attempt {attempt_counter}/{self.gw_max_injection_attempts}: Network SNR {self.network_snr:.2f} does not pass threshold [{self.gw_SNR_threshold_low}, {self.gw_SNR_threshold_high}]")

                if self.gw_load_existing_injection:
                    raise ValueError("SNR does not pass threshold, but loading existing injection. This should not happen!")

                if attempt_counter >= self.gw_max_injection_attempts:
                    raise RuntimeError(
                        f"Failed to generate injection parameters that meet SNR bounds after {self.gw_max_injection_attempts} attempts. "
                        f"SNR threshold: [{self.gw_SNR_threshold_low}, {self.gw_SNR_threshold_high}]. "
                        f"Last attempt network SNR: {self.network_snr:.2f}. "
                        f"Consider widening the prior ranges or adjusting SNR thresholds."
                    )

                logger.info("Resampling injection parameters...")

        logger.info(f"Network SNR passes threshold")

        # Add SNR and detector info to the UNTRANSFORMED injection for plotting
        # (The transformed 'injection' was used for waveform evaluation above,
        #  but we want to save the untransformed version with original parameters like 'q')
        injection_for_plotting.update(snr_dict)
        injection_for_plotting["network_SNR"] = self.network_snr

        # Also add detector etc info
        self.detector_param["duration"] = self.duration
        injection_for_plotting.update(self.detector_param)

        return injection_for_plotting
    
    def set_overlapping_gw_injection(self):
        """
        Function that creates an overlapping GW injection, taking into account the given priors and the SNR thresholds.

        TODO: CRITICAL - UNTESTED! This feature has NOT been tested with the new jim API!

              KNOWN ISSUES:
              1. Assumes exactly two signals - not general
              2. Massive code duplication with set_gw_injection() - violates DRY
              3. inject_signal() and add_signal() API may have changed in new jim
              4. No clear documentation on parameter naming convention (_1, _2 suffixes)
              5. EOS injection for overlapping BNS is NotImplementedError

              ACTION REQUIRED:
              1. TEST this method end-to-end with example_2 or similar
              2. Refactor to share code with set_gw_injection() (extract common logic)
              3. Update inject_signal() and add_signal() calls if jim API changed
              4. Add validation for overlapping prior structure
              5. Document expected prior parameter names

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Setting up overlapping GW injection . . . ")
        logger.info(f"The SNR thresholds are: {self.gw_SNR_threshold_low} - {self.gw_SNR_threshold_high}")
        logger.info(f"Maximum injection attempts allowed: {self.gw_max_injection_attempts}")
        pass_threshold = False
        attempt_counter = 0
        config_duration = eval(self.config["duration"])

        sample_key = jax.random.PRNGKey(self.seed)
        while not pass_threshold:
            
            # Generate the parameters or load them from an existing file
            injection_path = os.path.join(self.outdir, "injection.json")
            if self.gw_load_existing_injection:
                logger.info(f"Loading existing injection, path: {injection_path}")
                injection = json.load(open(injection_path))
            else:
                logger.info(f"Generating new injection")
                sample_key, subkey = jax.random.split(sample_key)
                injection = utils.generate_injection(injection_path, self.complete_prior, subkey)
            
            # TODO: here is where we might have to transform from prior to ripple/jim parameters
            
            # If a BNS run, we can infer Lambdas from a given EOS if desired and override the parameters
            if self.is_BNS_run and self.eos_file is not None:
                raise NotImplementedError("Overlapping BNS injections with EOS not implemented yet")
                logger.info(f"Computing lambdas from EOS file {self.eos_file} . . . ")
                injection = utils.inject_lambdas_from_eos(injection, self.eos_file)
            
            # Get duration based on Mc and fmin if not specified
            if self.config_duration is None:
                lower_M_c = min(injection["M_c_1"], injection["M_c_2"])
                duration = utils.signal_duration(self.fmin, lower_M_c)
                duration = 2 ** np.ceil(np.log2(duration))
                duration = float(duration)
                logger.info(f"Duration is not specified in the config. Computed chirp time: for fmin = {self.fmin} and M_c = {lower_M_c} is {duration}")
            else:
                duration = self.config_duration
                logger.info(f"Duration is specified in the config: {duration}")
                
            self.duration = duration

            # DON'T manually construct frequencies - let jim's Data objects handle this properly
            # Jim uses rfftfreq() which is numerically stable, while arange() has precision issues
            # self.frequencies will be set from detector data after inject_signal creates it
            
            # Make any necessary conversions
            # FIXME: hacky way for now --  if users specify iota in the injection, but sample over cos_iota and do the tfo, this breaks
            # try:
            injection = self.apply_transforms(injection)
            # except Exception as e:
            #     logger.error(f"Error in applying transforms: {e}")
            #     # raise ValueError("Error in applying transforms")
            
            # Setup the timing setting for the injection
            self.epoch = self.duration - self.post_trigger_duration
            self.gmst = Time(self.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
            
            # Get the array of the injection parameters
            # TODO: required_keys attribute removed in new jim API - need to implement waveform parameter mapping
            # For now, extracting all _1 and _2 suffixed parameters
            true_param_1 = {key[:-2]: float(injection[key]) for key in injection.keys() if key.endswith("_1")}
            true_param_2 = {key[:-2]: float(injection[key]) for key in injection.keys() if key.endswith("_2")}
            
            logger.info(f"The trial injection parameters are {injection}")
            
            self.detector_param_1 = {
                'psi':    injection["psi_1"],
                't_c':    injection["t_c_1"],
                'ra':     injection["ra_1"],
                'dec':    injection["dec_1"],
                'epoch':  self.epoch,
                'gmst':   self.gmst,
                }
            
            self.detector_param_2 = {
                'psi':    injection["psi_2"],
                't_c':    injection["t_c_2"],
                'ra':     injection["ra_2"],
                'dec':    injection["dec_2"],
                'epoch':  self.epoch,
                'gmst':   self.gmst,
                }
            
            # Generating the geocenter waveform
            snr_dict = {}
            logger.info("Injecting signals . . .")
            self.h_sky_1 = self.waveform(self.ifos[0].sliced_frequencies, true_param_1)
            self.h_sky_2 = self.waveform(self.ifos[0].sliced_frequencies, true_param_2)
            key = jax.random.PRNGKey(self.seed)
            logger.info("self.ifos")
            logger.info(self.ifos)
            for ifo in self.ifos:
                key, subkey = jax.random.split(key)
                ifo.inject_signal(
                    subkey,
                    self.frequencies,
                    self.h_sky_1,
                    self.detector_param_1,
                    psd_file=self.psds_dict[ifo.name]
                )
                
                ifo.add_signal(
                    self.h_sky_2,
                    self.detector_param_2,
                )
                
                # TODO: remove once tested
                logger.info(f"Signal injected in ifo {ifo.name}. Frequencies, data, and PSD:")
                logger.info(ifo.frequencies)
                logger.info(ifo.data)
                logger.info(ifo.psd)
                
                # Compute the SNR
                snr = utils.compute_snr(ifo, self.h_sky_1, self.detector_param_1)
                snr_dict[f"{ifo.name}_SNR_1"] = snr
                logger.info(f"SNR for ifo {ifo.name} and signal 1 is {snr}")
                
                snr = utils.compute_snr(ifo, self.h_sky_2, self.detector_param_2)
                snr_dict[f"{ifo.name}_SNR_2"] = snr
                logger.info(f"SNR for ifo {ifo.name} and signal 2 is {snr}")
            
            snr_list = list(snr_dict.values())
            self.network_snr = float(jnp.sqrt(jnp.sum(jnp.array(snr_list) ** 2)))
            
            logger.info(f"The network SNR is {self.network_snr}")

            # If the SNR is too low, we need to generate new parameters
            pass_threshold = self.network_snr > self.gw_SNR_threshold_low and self.network_snr < self.gw_SNR_threshold_high
            if not pass_threshold:
                attempt_counter += 1
                logger.info(f"Attempt {attempt_counter}/{self.gw_max_injection_attempts}: Network SNR {self.network_snr:.2f} does not pass threshold [{self.gw_SNR_threshold_low}, {self.gw_SNR_threshold_high}]")

                if self.gw_load_existing_injection:
                    raise ValueError("SNR does not pass threshold, but loading existing injection. This should not happen!")

                if attempt_counter >= self.gw_max_injection_attempts:
                    raise RuntimeError(
                        f"Failed to generate overlapping injection parameters that meet SNR bounds after {self.gw_max_injection_attempts} attempts. "
                        f"SNR threshold: [{self.gw_SNR_threshold_low}, {self.gw_SNR_threshold_high}]. "
                        f"Last attempt network SNR: {self.network_snr:.2f}. "
                        f"Consider widening the prior ranges or adjusting SNR thresholds."
                    )

                logger.info("Resampling overlapping injection parameters...")

        logger.info(f"Network SNR passes threshold")
        injection.update(snr_dict)
        injection["network_SNR"] = self.network_snr

        # Also add detector etc info
        self.detector_param_1["duration"] = self.duration
        self.detector_param_2["duration"] = self.duration
        
        injection.update(self.detector_param_1)
        injection.update(self.detector_param_2)
        
        return injection
    
    def apply_transforms(self, params: dict):
        """Apply likelihood transforms to injection parameters

        TODO: NEEDS IMPROVEMENT:
              1. Add validation that transform.forward() doesn't introduce NaNs/Infs
              2. Log which transforms are being applied and their effects
              3. Consider adding option to skip transforms for debugging
              4. Handle errors gracefully (some transforms may fail for edge cases)
        """
        for transform in self.likelihood_transforms:
            params = transform.forward(params)
        return params
    
    def dump_gw_injection(self):
        logger.info("Sanity checking the GW injection for ArrayImpl")
        for key, value in self.gw_injection.items():
            logger.info(f"   {key}: {value}")
        
        with open(os.path.join(self.outdir, "injection.json"), "w") as f:
            json.dump(self.gw_injection, f, indent=4, cls=utils.CustomJSONEncoder)
    
    def set_gw_data_from_npz(self):
        # TODO: Move this kind of functionality inside Jim where we then also check for consistency between the detectors
        for ifo in self.ifos:
            data_filename = eval(f"self.data_file_{ifo.name}")
            datadump = np.load(data_filename)
            
            frequencies = jnp.array(datadump["frequencies"])
            data = jnp.array(datadump["data"])
            psd = jnp.array(datadump["psd"])
            
            # Set the duration as well # TODO: has to be checked
            if self.config_duration is None or self.config_duration <= 0 or self.config_duration == "None":
                df = frequencies[1] - frequencies[0]
                self.duration = 1. / df
                self.duration = 2 ** np.ceil(np.log2(self.duration))
            else:
                self.duration = self.config_duration
            
            assert jnp.shape(frequencies) == jnp.shape(data), "Frequencies and data do not have the same shape."
            assert jnp.shape(frequencies) == jnp.shape(psd), "Frequencies and PSD do not have the same shape."
            
            ifo.frequencies = frequencies
            ifo.data = data
            ifo.psd = psd
    
    def dump_gw_data(self) -> None:
        # Dump the GW data
        logger.info("Dumping the GW data to npz files:")
        for ifo in self.ifos:
            ifo_path = os.path.join(self.outdir, f"{ifo.name}.npz")
            logger.info(f"    Dumping to {ifo_path}")
            np.savez(ifo_path, frequencies=ifo.frequencies, data=ifo.data, psd=ifo.psd)

    def set_waveform(self) -> Waveform:
        if self.waveform_approximant not in SUPPORTED_WAVEFORMS:
            raise ValueError(f"Waveform approximant {self.waveform_approximant} not supported. Supported waveforms are {SUPPORTED_WAVEFORMS}.")
        waveform_fn = WAVEFORMS_DICT[self.waveform_approximant]
        waveform = waveform_fn(f_ref = self.fref)
        return waveform

    def set_reference_waveform(self) -> Waveform:
        if self.waveform_approximant == "IMRPhenomD_NRTidalv2":
            logger.info("Using IMRPhenomD_NRTidalv2 waveform as reference waveform for the likelihood if relative binning is used")
            reference_waveform = RippleIMRPhenomD_NRTidalv2
        else:
            reference_waveform = WAVEFORMS_DICT[self.waveform_approximant]
        reference_waveform = reference_waveform(f_ref = self.fref)
        return reference_waveform
    
    def set_ifos(self) -> list[Detector]:
        # Go from string to list of ifos using factory functions
        detector_factory = {
            "H1": get_H1,
            "L1": get_L1,
            "V1": get_V1,
            "ET": get_ET,  # Returns list of 3 detectors (ET1, ET2, ET3)
            "CE": get_CE,
        }

        self.ifos_str: list[str] = self.config["ifos"].split(",")
        self.ifos_str = [x.strip() for x in self.ifos_str]

        ifos: list[Detector] = []
        for single_ifo_str in self.ifos_str:
            if single_ifo_str not in detector_factory:
                raise ValueError(
                    f"IFO {single_ifo_str} not supported. "
                    f"Supported IFOs are {list(detector_factory.keys())}."
                )
            # Call the factory function to create the detector
            new_ifo = detector_factory[single_ifo_str]()

            # Special handling for ET: get_ET() returns a list of 3 detectors
            if single_ifo_str == "ET":
                if isinstance(new_ifo, list):
                    ifos.extend(new_ifo)  # Add all 3 ET detectors (ET1, ET2, ET3)
                else:
                    ifos.append(new_ifo)
            else:
                ifos.append(new_ifo)
        return ifos
    
    