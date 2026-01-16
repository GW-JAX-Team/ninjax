"""Utility functions for ninjax pipes

TODO: NEEDS ORGANIZATION - This file contains many different utilities that could be split:

      POTENTIAL IMPROVEMENTS:
      1. Split into separate modules:
         - plotting.py (plot_*, corner plot utils)
         - io_utils.py (directory checks, JSON encoder)
         - physics_utils.py (signal_duration, compute_snr, Mc_eta_to_ms wrappers)
         - injection_utils.py (generate_injection, inject_lambdas_from_eos)

      2. Add comprehensive docstrings with examples

      3. Add input validation (check for NaNs, infinities, negative values where inappropriate)

      4. Make plotting more configurable (DPI, figure sizes, color schemes)

      TESTING NEEDED:
      - Test compute_snr with different detector configurations
      - Test generate_injection with edge cases (JAX arrays, regular arrays)
      - Test inject_lambdas_from_eos with different EOS files
      - Add unit tests for all functions
"""

import logging
import os
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.likelihood import SingleEventLikelihood, HeterodynedTransientLikelihoodFD
from jimgw.core.prior import CombinePrior
import corner
import jax
from jaxtyping import Array, Float

from ripplegw import Mc_eta_to_ms

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

matplotlib_params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(matplotlib_params)

LABELS_TRANSLATION_DICT = {"M_c": r'$M_c/M_\odot$', 
                           "q": r'$q$', 
                           "s1_z": r'$\chi_1$',
                           "s2_z": r'$\chi_2$',
                           "lambda_1": r'$\Lambda_1$',
                           "lambda_2": r'$\Lambda_2$',
                           "f_stop": r'$f_{\rm{stop}}$',
                           "a_1": r'$a_1$',
                           "a_2": r'$a_2$',
                           "d_L": r'$d_{\rm{L}}/{\rm Mpc}$',
                           "t_c": r'$t_c$', 
                           "phase_c": r'$\phi_c$', 
                           "iota": r'$\iota$', 
                           "psi": r'$\psi$',
                           "ra": r'$\alpha$',
                           "dec": r'$\delta$'}

def check_directory_exists_and_if_not_mkdir(directory):
    """Checks if the given directory exists and creates it if it does not exist

    Parameters
    ----------
    directory: str
        Name of the directory

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Making directory {directory}")
    else:
        logger.debug(f"Directory {directory} exists")
        
class DuplicateErrorDict(dict):
    """An dictionary with immutable key-value pairs

    Once a key-value pair is initialized, any attempt to update the value for
    an existing key will raise a ValueError.

    Raises
    ------
    ValueError:
        When a user attempts to update an existing key.

    """

    def __init__(self, color=True, *args):
        dict.__init__(self, args)
        self.color = color

    def __setitem__(self, key, val):
        if key in self:
            msg = f"Your ini file contains duplicate '{key}' keys"
            raise ValueError(msg)
        dict.__setitem__(self, key, val)

class CustomJSONEncoder(json.JSONEncoder):
    """A JSON encoder that can deal with JAX arrays"""
    def default(self, obj):
        # Handle JAX arrays
        if isinstance(obj, jnp.ndarray):
            return np.array(obj).tolist()
        # Handle other custom cases if necessary
        return super().default(obj)

def setup_logger(outdir=None, label=None, log_level="INFO"):
    """Setup logging output: call at the start of the script to use

    Parameters
    ----------
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if "-v" in sys.argv or "--verbose" in sys.argv:
        log_level = "DEBUG"

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError(f"log_level {log_level} not understood")
    else:
        level = int(log_level)

    logger = logging.getLogger("ninjax")
    logger.propagate = False
    logger.setLevel(level)

    streams = [isinstance(h, logging.StreamHandler) for h in logger.handlers]
    if len(streams) == 0 or not all(streams):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
            )
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([isinstance(h, logging.FileHandler) for h in logger.handlers]) is False:
        if label:
            if outdir:
                check_directory_exists_and_if_not_mkdir(outdir)
            else:
                outdir = "."
            log_file = f"{outdir}/{label}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
                )
            )

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)
        
setup_logger()
logger = logging.getLogger("ninjax")


###################
### Auxiliaries ###
###################

def signal_duration(f: float, chirp_mass: float):
    """
    f: frequency in HZ, e.g. starting frequency
    M: chirp mass in solar mass
    returns: tau in s
    """
    c = 3*10**8                     #m/s
    G = 6.67*10**(-11)              #m^3 kg^-1 s^-2
    solar_mass = 1.9891*10**30      #kg
    chirp_mass = chirp_mass * solar_mass
    return (5/256) * ((c**5)/G**(5/3)) * (((np.pi * f)**(-8/3)) / chirp_mass**(5/3))

def compute_snr(detector: Detector, h_sky: dict, detector_params: dict):
    """Compute the SNR of an event for a single detector, given the waveform generated in the sky.

    Args:
        detector (Detector): Detector object from jim.
        h_sky (dict): Dict of jax numpy array containing the waveform strain as a function of frequency in the sky frame
        detector_params (dict): Dictionary containing parameters of the event relevant for the detector.
    """
    # Use sliced frequencies and PSD to respect frequency bounds (fmin, fmax)
    frequencies = detector.sliced_frequencies
    df = frequencies[1] - frequencies[0]
    align_time = jnp.exp(
        -1j * 2 * jnp.pi * frequencies * (detector_params["epoch"] + detector_params["t_c"])
    )

    # Generate waveform on sliced frequencies and apply frequency mask
    waveform_dec = (
                detector.fd_response(frequencies, h_sky, detector_params) * align_time
            )

    snr = 4 * jnp.sum(jnp.conj(waveform_dec) * waveform_dec / detector.sliced_psd * df).real
    snr = float(jnp.sqrt(snr))
    return snr

def generate_params_dict(prior_low: jnp.array, prior_high: jnp.array, params_names: dict) -> dict:
    """
    Generate a dictionary of parameters from the prior range.

    Args:
        prior_low (jnp.array): Lower bound of the priors
        prior_high (jnp.array): Upper bound of the priors
        params_names (dict): Names of the parameters

    Returns:
        dict: Dictionary of key-value pairs of the parameters
    """
    params_dict = {}
    for low, high, param in zip(prior_low, prior_high, params_names):
        params_dict[param] = np.random.uniform(low, high)
    return params_dict

def generate_injection(config_path: str,
                       prior: CombinePrior,
                       sample_key) -> dict:
    """
    From a given prior range and parameter names, generate the injection parameters

    TODO: IMPROVEMENTS:
          1. config_path argument is not used! Either use it or remove it
          2. Add validation that all values are finite (not NaN/inf)
          3. Consider adding option to save injection to file directly
          4. Better error handling for JAX array conversion
    """

    # Generate parameters
    params_sampled = prior.sample(sample_key, 1)
    # TODO: FRAGILE - This JAX array conversion could fail for some array types
    #       Consider using jnp.asarray first for consistency
    # Convert JAX arrays to Python floats using .item() for scalar extraction
    params_dict = {key: float(value.item()) if hasattr(value, 'item') else float(value[0]) for key, value in params_sampled.items()}

    logger.info("Sanity check: generated parameters:")
    logger.info(params_dict)

    return params_dict

def inject_lambdas_from_eos(injection: dict, lambdas_eos_file: str):
    
    Mc, q = injection['M_c'], injection['q']
    eta = q / (1 + q)**2
    eos = np.load(lambdas_eos_file)
    masses, Lambdas = eos['masses_EOS'], eos['Lambdas_EOS']
    m1, m2 = Mc_eta_to_ms(jnp.array([Mc, eta]))
    
    # Use float since sometimes jnp.arrays cause weird behavior
    m1 = float(m1)
    m2 = float(m2)
    
    lambda_1 = np.interp(m1, masses, Lambdas)
    lambda_2 = np.interp(m2, masses, Lambdas)
    
    injection['lambda_1'] = lambda_1
    injection['lambda_2'] = lambda_2
    
    logger.info(f"Injected lambda_1: {lambda_1}")
    logger.info(f"Injected lambda_2: {lambda_2}")
    
    return injection

###
### PLOTTING ###
###

def plot_accs(accs, label, name, outdir):
    
    eps = 1e-3
    plt.figure(figsize=(10, 6))
    plt.plot(accs, label=label)
    plt.ylim(0 - eps, 1 + eps)
    
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()
    
def plot_log_prob(log_prob, label, name, outdir):
    log_prob = np.mean(log_prob, axis = 0)
    plt.figure(figsize=(10, 6))
    plt.plot(log_prob, label=label)
    # plt.yscale('log')
    plt.ylabel(label)
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()

    
def plot_chains(chains: jnp.array, 
                name: str, 
                outdir: str, 
                labels: list[str],
                truths: np.array = None):
    
    chains = np.asarray(chains)
    fig = corner.corner(chains, labels = labels, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
    fig.savefig(f"{outdir}{name}.png", bbox_inches='tight')
    plt.close()
    
# def plot_chains_from_file(outdir, load_true_params: bool = False):
    
#     filename = outdir + 'results_production.npz'
#     data = np.load(filename)
#     chains = data['chains']
#     my_chains = []
#     n_dim = np.shape(chains)[-1]
#     for i in range(n_dim):
#         values = chains[:, :, i].flatten()
#         my_chains.append(values)
#     my_chains = np.array(my_chains).T
#     chains = chains.reshape(-1, 13)
#     if load_true_params:
#         truths = load_true_params_from_config(outdir)
#     else:
#         truths = None
    
#     plot_chains(chains, truths, 'results', outdir)
    
def plot_accs_from_file(outdir):
    
    filename = outdir + 'results_production.npz'
    data = np.load(filename)
    local_accs = data['local_accs']
    global_accs = data['global_accs']
    
    local_accs = np.mean(local_accs, axis = 0)
    global_accs = np.mean(global_accs, axis = 0)
    
    plot_accs(local_accs, 'local_accs', 'local_accs_production', outdir)
    plot_accs(global_accs, 'global_accs', 'global_accs_production', outdir)
    
def plot_log_prob_from_file(outdir, which_list = ['training', 'production']):
    
    for which in which_list:
        filename = outdir + f'results_{which}.npz'
        data = np.load(filename)
        log_prob= data['log_prob']
        plot_log_prob(log_prob, f'log_prob_{which}', f'log_prob_{which}', outdir)
    
    
# def load_true_params_from_config(outdir):
    
#     config = outdir + 'config.json'
#     # Load the config   
#     with open(config) as f:
#         config = json.load(f)
#     true_params = np.array([config[key] for key in NAMING])
    
#     # Convert cos_iota and sin_dec to iota and dec
#     cos_iota_index = NAMING.index('cos_iota')
#     sin_dec_index = NAMING.index('sin_dec')
#     true_params[cos_iota_index] = np.arccos(true_params[cos_iota_index])
#     true_params[sin_dec_index] = np.arcsin(true_params[sin_dec_index])
    
#     return true_params

def plot_loss_vals(loss_values, label, name, outdir):
    loss_values = loss_values.reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label=label)
    
    plt.ylabel(label)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.savefig(f"{outdir}{name}.png", bbox_inches='tight')  
    plt.close()