import os
import json
from typing import Callable
import numpy as np
from astropy.time import Time
import jax 
import jax.numpy as jnp

from jimgw.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD_NRTidalv2_no_taper, RippleIMRPhenomD, RippleIMRPhenomPv2
from jimgw.single_event.detector import Detector, TriangularNetwork2G, H1, L1, V1, ET, CE
from jimgw.prior import Composite

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger

WAVEFORMS_DICT = {"TaylorF2": RippleTaylorF2, 
                  "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
                  "IMRPhenomD": RippleIMRPhenomD,
                  "IMRPhenomPv2": RippleIMRPhenomPv2,
                  }

SUPPORTED_WAVEFORMS = list(WAVEFORMS_DICT.keys())
BNS_WAVEFORMS = ["IMRPhenomD_NRTidalv2"]

class GWPipe:
    def __init__(self, 
                 config: dict, 
                 outdir: str, 
                 prior: Composite,
                 generation_prior: Composite,
                 prior_bounds: np.array, 
                 seed: int,
                 transforms: list[Callable]):
        self.config = config
        self.outdir = outdir
        self.complete_prior = prior
        self.generation_prior = generation_prior
        self.complete_prior_bounds = prior_bounds
        self.seed = seed
        self.injection_generation_seed = int(config["injection_generation_seed"])
        self.transforms = transforms
        
        # Initialize other GW-specific attributes
        self.eos_file = self.set_eos_file()
        self.is_BNS_run = (self.waveform_approximant_1 in BNS_WAVEFORMS) or (self.waveform_approximant_2 in BNS_WAVEFORMS)
        logger.info(f"self.is_BNS_run is {self.is_BNS_run}")
        self.psds_dict = self.set_psds_dict()
        self.ifos = self.set_ifos()
        if self.config["gw_is_overlapping"]:
            # Set the waveforms separately
            self.waveform_1 = self.set_waveform_1()
            self.reference_waveform_1 = self.set_reference_waveform_1()
            
            self.waveform_2 = self.set_waveform_2()
            self.reference_waveform_2 = self.set_reference_waveform_2()
            
        else:
            self.waveform = self.set_waveform()
            self.reference_waveform = self.set_reference_waveform()
        
        # TODO: data loading if preprocesse data is shared
        # Check if an injection and if has to be loaded, or if provided GW data must be loaded
        self.is_gw_injection = eval(self.config["gw_injection"])
        if self.is_gw_injection:
            logger.info(f"GW run is an injection")
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
    def gw_dump_injection_data(self):
        return eval(self.config["gw_dump_injection_data"])
    
    @property
    def gw_SNR_threshold_low(self):
        return float(self.config["gw_SNR_threshold_low"])

    @property
    def gw_SNR_threshold_high(self):
        return float(self.config["gw_SNR_threshold_high"])
    
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
    def waveform_approximant_1(self):
        if self.config["waveform_approximant_1"].lower() == "none":
            logger.info(f"Note: No specific waveform approximant 1 so defaulting to {self.config['waveform_approximant']}")
            return self.config["waveform_approximant"]
        else:
            return self.config["waveform_approximant_1"]
        
    @property
    def waveform_approximant_2(self):
        if self.config["waveform_approximant_2"].lower() == "none":
            logger.info(f"Note: No specific waveform approximant 2 so defaulting to {self.config['waveform_approximant']}")
            return self.config["waveform_approximant"]
        else:
            return self.config["waveform_approximant_2"]
    
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
        pass_threshold = False
        
        sample_key = jax.random.PRNGKey(self.injection_generation_seed)
        while not pass_threshold:
            
            # Generate the parameters or load them from an existing file
            injection_path = os.path.join(self.outdir, "injection.json")
            if self.gw_load_existing_injection:
                logger.info(f"Loading existing injection, path: {injection_path}")
                injection = json.load(open(injection_path))
            else:
                logger.info(f"Generating new injection")
                sample_key, subkey = jax.random.split(sample_key)
                injection = utils.generate_injection(injection_path, self.generation_prior, subkey)
            
            # If a BNS run, we can infer Lambdas from a given EOS if desired and override the parameters
            if self.is_BNS_run and self.eos_file is not None:
                logger.info(f"Computing lambdas from EOS file {self.eos_file} . . . ")
                injection = utils.inject_lambdas_from_eos(injection, self.eos_file)
                lambdas_set = True
            else:
                lambdas_set = False
            
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
                
            # Construct frequencies array
            self.frequencies = jnp.arange(
                self.fmin,
                self.fmax,
                1. / self.duration
            )
            
            # Make any necessary conversions
            # FIXME: hacky way for now --  if users specify iota in the injection, but sample over cos_iota and do the tfo, this breaks
            try:
                # If lambdas are already set from EOS file, save them and overwrite later on
                if lambdas_set:
                    original_lambda_1 = injection["lambda_1"]
                    original_lambda_2 = injection["lambda_2"]
                injection = self.apply_transforms(injection)
                
                # Now overwrite with original lambdas
                if lambdas_set:
                    injection["lambda_1"] = original_lambda_1
                    injection["lambda_2"] = original_lambda_2
            except Exception as e:
                logger.error(f"Error in applying transforms: {e}")
            
            logger.info("After transforms, the injection parameters are:")
            logger.info(injection)
            
            # Setup the timing setting for the injection
            self.epoch = self.duration - self.post_trigger_duration
            self.gmst = Time(self.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
            
            # Get the array of the injection parameters
            true_param = {key: float(injection[key]) for key in self.waveform.required_keys + ["t_c", "psi", "ra", "dec"]}
            
            logger.info(f"The trial injection parameters are {true_param}")
            
            self.detector_param = {
                'psi':    injection["psi"],
                't_c':    injection["t_c"],
                'ra':     injection["ra"],
                'dec':    injection["dec"],
                'epoch':  self.epoch,
                'gmst':   self.gmst,
                }
            
            # Generating the geocenter waveform
            logger.info("Injecting signals . . .")
            self.h_sky = self.waveform(self.frequencies, true_param)
            key = jax.random.PRNGKey(self.seed)
            logger.info("self.ifos")
            logger.info(self.ifos)
            for ifo in self.ifos:
                key, subkey = jax.random.split(key)
                ifo.inject_signal(
                    subkey,
                    self.frequencies,
                    self.h_sky,
                    self.detector_param,
                    psd_file=self.psds_dict[ifo.name]
                )
                
                # TODO: remove once tested
                logger.info(f"Signal injected in ifo {ifo.name}. Frequencies, data, and PSD:")
                logger.info(ifo.frequencies)
                logger.info(ifo.data)
                logger.info(ifo.psd)
            
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
                if self.gw_load_existing_injection:
                    raise ValueError("SNR does not pass threshold, but loading existing injection. This should not happen!")
                else:
                    logger.info("The network SNR does not pass the threshold, trying again")
                    
        logger.info(f"Network SNR passes threshold")
        injection.update(snr_dict)
        injection["network_SNR"] = self.network_snr
        
        # Also add detector etc info
        self.detector_param["duration"] = self.duration
        injection.update(self.detector_param)
        
        return injection
    
    def set_overlapping_gw_injection(self):
        """
        Function that creates an overlapping GW injection, taking into account the given priors and the SNR thresholds.
        # TODO: assuming exactly two signals for now, is that OK?
        # TODO: make sure there is as minimal code duplication with the single injection function as possible
        # TODO: do not hardcode injection.json, make more flexible

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Setting up overlapping GW injection . . . ")
        logger.info(f"The SNR thresholds are: {self.gw_SNR_threshold_low} - {self.gw_SNR_threshold_high}")
        pass_threshold = False
        
        sample_key = jax.random.PRNGKey(self.injection_generation_seed)
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
                logger.info(f"Computing lambdas from EOS file {self.eos_file} . . . ")
                injection = utils.inject_lambdas_from_eos_overlapping(injection, self.eos_file)
            
            # Get duration based on Mc and fmin if not specified
            if self.config_duration is None:
                lower_M_c = min(injection["M_c_1"], injection["M_c_2"])
                duration = utils.signal_duration(self.fmin, lower_M_c)
                duration = 2 ** np.ceil(np.log2(duration))
                duration = float(duration)
                logger.info(f"Duration is not specified in the config. Computed chirp time: for fmin = {self.fmin} and M_c = {lower_M_c} is {duration}")
                if duration < 4.0:
                    logger.info(f"Duration computed is too short, setting to default of 4.0s")
                    duration = 4.0
            else:
                duration = self.config_duration
                logger.info(f"Duration is specified in the config: {duration}")
                
            self.duration = duration
                
            # Construct frequencies array
            self.frequencies = jnp.arange(
                self.fmin,
                self.fmax,
                1. / self.duration
            )
            
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
            required_keys_1 = [f"{k}_1" for k in self.waveform_1.required_keys]
            required_keys_2 = [f"{k}_2" for k in self.waveform_2.required_keys]
            
            true_param_1 = {key[:-2]: float(injection[key]) for key in required_keys_1 + ["t_c_1", "psi_1", "ra_1", "dec_1"]}
            true_param_2 = {key[:-2]: float(injection[key]) for key in required_keys_2 + ["t_c_2", "psi_2", "ra_2", "dec_2"]}
            
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
            self.h_sky_1 = self.waveform_1(self.frequencies, true_param_1)
            self.h_sky_2 = self.waveform_2(self.frequencies, true_param_2)
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
                    psd_file=self.psds_dict[ifo.name],
                    dump_data=self.gw_dump_injection_data,
                    outdir=self.outdir
                )
                
                ifo.add_signal(
                    self.h_sky_2,
                    self.detector_param_2,
                    dump_data=self.gw_dump_injection_data,
                    outdir=self.outdir
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
            
            # Comppute the network SNR for each signal separately:
            snr_dict_signal_1 = [snr_dict[f"{ifo.name}_SNR_1"] for ifo in self.ifos]
            snr_dict_signal_2 = [snr_dict[f"{ifo.name}_SNR_2"] for ifo in self.ifos]
            
            network_snr_1 = float(jnp.sqrt(jnp.sum(jnp.array(snr_dict_signal_1) ** 2)))
            network_snr_2 = float(jnp.sqrt(jnp.sum(jnp.array(snr_dict_signal_2) ** 2)))
            self.network_snr = np.sqrt(network_snr_1 ** 2 + network_snr_2 ** 2)
            
            logger.info(f"The network SNR of signal 1 is {network_snr_1}")
            logger.info(f"The network SNR of signal 2 is {network_snr_2}")
            
            # If the SNR is too low, we need to generate new parameters
            pass_threshold_1 = (network_snr_1 > self.gw_SNR_threshold_low) and (network_snr_1 < self.gw_SNR_threshold_high)
            pass_threshold_2 = (network_snr_2 > self.gw_SNR_threshold_low) and (network_snr_2 < self.gw_SNR_threshold_high)
            
            pass_threshold = pass_threshold_1 and pass_threshold_2
            
            if not pass_threshold:
                if self.gw_load_existing_injection:
                    raise ValueError("SNR does not pass threshold, but loading existing injection. This should not happen!")
                else:
                    logger.info("The network SNR does not pass the threshold, trying again")
                    
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
        for transform in self.transforms:
            params = transform(params)
        return params
    
    def dump_gw_injection(self):
        logger.info("Sanity checking the GW injection for ArrayImpl")
        for key, value in self.gw_injection.items():
            logger.info(f"   {key}: {value}")
        
        with open(os.path.join(self.outdir, "injection.json"), "w") as f:
            json.dump(self.gw_injection, f, indent=4, cls=utils.CustomJSONEncoder)
    
    def set_gw_data_from_npz(self):
        logger.info("We are setting the ifo data from pregenerated npz files")
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
            
            logger.info(f"Checking the data loaded from npz files for ifo = {ifo.name}:")
            logger.info(f"    frequencies: {frequencies}")
            logger.info(f"    data: {data}")
            logger.info(f"    psd: {psd}")
            
            # Check for NaNs or infs
            if jnp.any(jnp.isnan(data)):
                logger.info(f"Data for {ifo.name} contains NaNs")
            if jnp.any(jnp.isinf(data)):
                logger.info(f"Data for {ifo.name} contains infs")
            if jnp.any(jnp.isnan(psd)):
                logger.info(f"PSD for {ifo.name} contains NaNs")
            if jnp.any(jnp.isinf(psd)):
                logger.info(f"PSD for {ifo.name} contains infs")
            ifo.data = jnp.nan_to_num(data)
            ifo.psd = jnp.nan_to_num(psd)
            ifo.frequencies = jnp.nan_to_num(frequencies)
            
            
            
    
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
    
    # TODO: clumsy, need to change this or make more general or something in the future
    def set_waveform_1(self) -> Waveform:
        if self.waveform_approximant_1 not in SUPPORTED_WAVEFORMS:
            raise ValueError(f"Waveform approximant {self.waveform_approximant_1} not supported. Supported waveforms are {SUPPORTED_WAVEFORMS}.")
        waveform_fn_1 = WAVEFORMS_DICT[self.waveform_approximant_1]
        waveform_1 = waveform_fn_1(f_ref = self.fref)
        return waveform_1
    
    def set_waveform_2(self) -> Waveform:
        if self.waveform_approximant_2 not in SUPPORTED_WAVEFORMS:
            raise ValueError(f"Waveform approximant {self.waveform_approximant_2} not supported. Supported waveforms are {SUPPORTED_WAVEFORMS}.")
        waveform_fn_2 = WAVEFORMS_DICT[self.waveform_approximant_2]
        waveform_2 = waveform_fn_2(f_ref = self.fref)
        return waveform_2

    def set_reference_waveform(self) -> Waveform:
        if self.waveform_approximant == "IMRPhenomD_NRTidalv2":
            logger.info("Using IMRPhenomD_NRTidalv2 waveform. Therefore, we will use no taper as the reference waveform for the likelihood if relative binning is used")
            reference_waveform = RippleIMRPhenomD_NRTidalv2_no_taper
        else:
            reference_waveform = WAVEFORMS_DICT[self.waveform_approximant]
        reference_waveform = reference_waveform(f_ref = self.fref)
        return reference_waveform
    
    def set_reference_waveform_1(self) -> Waveform:
        if self.waveform_approximant_1 == "IMRPhenomD_NRTidalv2":
            logger.info("Using IMRPhenomD_NRTidalv2 waveform for signal 1. Therefore, we will use no taper as the reference waveform for the likelihood if relative binning is used")
            reference_waveform_1 = RippleIMRPhenomD_NRTidalv2_no_taper
        else:
            reference_waveform_1 = WAVEFORMS_DICT[self.waveform_approximant_1]
        reference_waveform_1 = reference_waveform_1(f_ref = self.fref)
        return reference_waveform_1
    
    def set_reference_waveform_2(self) -> Waveform:
        if self.waveform_approximant_2 == "IMRPhenomD_NRTidalv2":
            logger.info("Using IMRPhenomD_NRTidalv2 waveform for signal 2. Therefore, we will use no taper as the reference waveform for the likelihood if relative binning is used")
            reference_waveform_2 = RippleIMRPhenomD_NRTidalv2_no_taper
        else:
            reference_waveform_2 = WAVEFORMS_DICT[self.waveform_approximant_2]
        reference_waveform_2 = reference_waveform_2(f_ref = self.fref)
        return reference_waveform_2
    
    def set_ifos(self) -> list[Detector]:
        # Go from string to list of ifos
        supported_ifos = ["H1", "L1", "V1", "ET", "CE"]
        self.ifos_str: list[str] = self.config["ifos"].split(",")
        self.ifos_str = [x.strip() for x in self.ifos_str]
        
        ifos: list[Detector] = []
        for single_ifo_str in self.ifos_str:
            if single_ifo_str not in supported_ifos:
                raise ValueError(f"IFO {single_ifo_str} not supported. Supported IFOs are {supported_ifos}.")
            new_ifo = eval(single_ifo_str)
            if isinstance(new_ifo, TriangularNetwork2G):
                ifos += new_ifo.ifos
            else:
                ifos.append(new_ifo)
        return ifos
    
    