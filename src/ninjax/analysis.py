"""Main analysis script - runs the full ninjax sampling pipeline

TODO: IMPROVEMENTS NEEDED:

      TESTED FEATURES (example_1):
      - Full sampling pipeline (training + production)
      - Results saving (npz files with chains, log_prob, accs)
      - Corner plotting with injection truth values
      - NF model saving and sampling
      - Polynomial learning rate scheduler

      AREAS FOR IMPROVEMENT:
      1. Better exception handling (currently bare try-except for corner plot)
      2. Add checkpoint/resume functionality for long runs
      3. Add real-time monitoring/progress tracking
      4. Configurable output formats (HDF5, Zarr, etc.)
      5. Add diagnostics (ESS, R-hat, etc.) to output
      6. Modularize body() function (too long, many responsibilities)
      7. Add validation of sampling results (check for NaNs, convergence)

      TESTING NEEDED:
      - Test with non-GW likelihoods
      - Test checkpoint/resume
      - Test with different scheduler configurations
      - Test error handling for failed sampling
"""

import os
import sys
import numpy as np
# Regular imports
import copy
import numpy as np
from astropy.time import Time
import time
import shutil
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import pandas as pd

from jimgw.core.jim import Jim

from jimgw.core.single_event.utils import C1_C2_to_f_stop, Mc_q_to_m1_m2, L1_L2_to_a1_a2

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger
from ninjax.pipes.ninjax_pipe import NinjaxPipe

from ninjax.blackjax_ns import (
    run_blackjax_ns_gw,
    setup_sample_transforms,
    create_logprior_fn,
    create_loglikelihood_fn,
    create_unit_cube_stepper
)

# Configure jim logger to output at DEBUG level (after all imports)
import logging
jim_logger = logging.getLogger("jim")
jim_logger.setLevel(logging.DEBUG)
jim_logger.propagate = False  # Prevent duplicate logs

# Clear any existing handlers and create new ones with same format but DEBUG level
jim_logger.handlers.clear()
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        # Create stream handler for jim with DEBUG level
        jim_stream_handler = logging.StreamHandler()
        jim_stream_handler.setLevel(logging.DEBUG)
        jim_stream_handler.setFormatter(handler.formatter)
        jim_logger.addHandler(jim_stream_handler)
    elif isinstance(handler, logging.FileHandler):
        # Create file handler for jim with DEBUG level
        jim_file_handler = logging.FileHandler(handler.baseFilename)
        jim_file_handler.setLevel(logging.DEBUG)
        jim_file_handler.setFormatter(handler.formatter)
        jim_logger.addHandler(jim_file_handler)

####################
### Script setup ###
####################

def get_sampler_state(jim, training: bool = False):
    """
    Extract sampler state from jim.sampler.resources.

    Args:
        jim: Jim instance after sampling
        training: If True, extract training state; if False, extract production state

    Returns:
        dict with keys: chains, log_prob, local_accs, global_accs, loss_vals (training only)

    TODO: FRAGILE - This assumes specific resource naming convention!
          If jim's internal resource structure changes, this will break
          Consider:
          1. Adding validation that resources exist before accessing
          2. Raising informative errors if resources are missing
          3. Making resource names configurable or documented
    """
    resources = jim.sampler.resources
    suffix = "_training" if training else "_production"

    state = {
        "chains": resources[f"positions{suffix}"].data,
        "log_prob": resources[f"log_prob{suffix}"].data,
        "local_accs": resources[f"local_accs{suffix}"].data,
        "global_accs": resources[f"global_accs{suffix}"].data,
    }

    if training:
        state["loss_vals"] = resources["loss_buffer"].data

    return state

def save_nf_model(jim, path: str):
    """
    Save the trained normalizing flow model.

    Args:
        jim: Jim instance after sampling
        path: Path to save model (without .eqx extension)
    """
    model = jim.sampler.resources["model"]
    model.save_model(path)

def sample_from_nf(jim, n_samples: int, rng_key):
    """
    Sample from the trained normalizing flow and transform back to prior space.

    Args:
        jim: Jim instance after sampling
        n_samples: Number of samples to generate
        rng_key: JAX random key

    Returns:
        dict of samples in prior parameter space
    """
    # Sample from the trained flow (returns raw samples in sampling space)
    nf_samples, _ = jim.sampler.resources["global_sampler"].sample_flow(rng_key, n_samples)

    # Convert to dictionary format
    nf_samples = jax.vmap(jim.add_name)(nf_samples)

    # Apply inverse sample transforms to get back to prior parameter space
    if jim.sample_transforms:
        for transform in reversed(jim.sample_transforms):
            nf_samples = jax.vmap(transform.backward)(nf_samples)

    return nf_samples

def body(pipe: NinjaxPipe):
    start_time = time.time()
    
    # Before main code, check if outdir is correct dir format
    outdir = pipe.outdir
    if outdir[-1] != "/":
        outdir += "/"
    logger.info(f"Saving output to {outdir}")
    
    jim_hyperparameters = pipe.jim_hyperparameters
    analysis_config = pipe.analysis_config

    # Handle mala_step_size_scale if provided
    if "mala_step_size_scale" in jim_hyperparameters:
        mala_step_size_scale = jim_hyperparameters.pop("mala_step_size_scale")
        base_step_size = jim_hyperparameters["mala_step_size"]

        # Create per-parameter step size array
        step_sizes = jnp.ones(pipe.n_dim) * base_step_size
        for param_name, scale in mala_step_size_scale.items():
            if param_name in pipe.naming:
                param_idx = pipe.naming.index(param_name)
                step_sizes = step_sizes.at[param_idx].set(base_step_size * scale)
                logger.info(f"Scaled step size for {param_name}: {base_step_size} * {scale} = {base_step_size * scale}")
            else:
                logger.warning(f"Parameter {param_name} in mala_step_size_scale not found in prior parameters: {pipe.naming}")

        jim_hyperparameters["mala_step_size"] = step_sizes
        logger.info(f"Final per-parameter step sizes: {step_sizes}")

    ### POLYNOMIAL SCHEDULER
    # TODO: move this to the pipe generation
    if analysis_config["use_scheduler"]:
        logger.info("Using polynomial learning rate scheduler")
        total_epochs = jim_hyperparameters["n_epochs"] * jim_hyperparameters["n_training_loops"]
        start = int(total_epochs / 10)
        start_lr = 1e-3
        end_lr = 1e-4
        power = 3.0
        schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)
        jim_hyperparameters["learning_rate"] = schedule_fn

    logger.info("The hyperparameters passed to Jim are:")
    for key, val in jim_hyperparameters.items():
        logger.info(f"   {key}: {val}")
    logger.info("Analysis configuration:")
    for key, val in analysis_config.items():
        logger.info(f"   {key}: {val}")
    
    def run_flowMC():
        # Create jim object
        jim = Jim(
            pipe.likelihood,
            pipe.complete_prior,
            sample_transforms=pipe.sample_transforms,
            likelihood_transforms=pipe.likelihood_transforms,
            rng_key=jax.random.PRNGKey(pipe.sampling_seed),
            **jim_hyperparameters
        )

        # Fetch injected values for the plotting below
        if pipe.gw_pipe.is_gw_injection:
            logger.info("Fetching the injected values for plotting")
            with open(os.path.join(pipe.outdir, "injection.json"), "r") as f:
                injection = json.load(f)
            #Transfrom C1 and C2 to f_stop if needed
            if "C_1" and "C_2" in injection:
                print("Transforming C1 and C2 to f_stop for injection values")
                m1, m2  = Mc_q_to_m1_m2(injection["M_c"], injection["q"])
                f_stop = C1_C2_to_f_stop(injection["C_1"], injection["C_2"], m1, m2)
                injection["f_stop"] = f_stop
                del injection["C_1"]
                del injection["C_2"]
            #Add a1 and a2 from Lambda1 and Lambda2 if needed
            #TODO: add extra check if TF2_SSM is used
            #if "a_1" and "a_2" not in injection:
            if (not pipe.complete_prior.has_param("a_1")) or (not pipe.complete_prior.has_param("a_2")):
                print("Calculating a1 and a2 from tidal deformability")
                a1, a2 = L1_L2_to_a1_a2(injection["lambda_1"], injection['lambda_2'])
                injection["a_1"] = a1
                injection["a_2"] = a2
            truths = np.array([injection[key] for key in pipe.keys_to_plot])        
        else:
            truths = None

        ### Finally, do the sampling
        jim.sample()

        # Plot training
        name = outdir + f'results_training.npz'
        logger.info(f"Saving training results to {name}")
        state = get_sampler_state(jim, training=True)
        chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
        
        local_accs = jnp.mean(local_accs, axis=0)
        global_accs = jnp.mean(global_accs, axis=0)
        if analysis_config["save_training_chains"]:
            np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals, chains=chains)
        else:
            np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals)
        
        utils.plot_accs(local_accs, "Local accs (training)", "local_accs_training", outdir)
        utils.plot_accs(global_accs, "Global accs (training)", "global_accs_training", outdir)
        utils.plot_loss_vals(loss_vals, "Loss", "loss_vals", outdir)
        utils.plot_log_prob(log_prob, "Log probability (training)", "log_prob_training", outdir)
        
        # Save the NF and also some samples from the flow
        logger.info("Saving the NF model")
        save_nf_model(jim, outdir + "nf_model")

        logger.info("Sampling from the trained NF")
        name = outdir + 'results_NF.npz'
        nf_chains = sample_from_nf(jim, 10_000, jax.random.PRNGKey(pipe.sampling_seed + 1))
        # Convert dict to arrays for saving
        nf_chains = {key: np.array(nf_chains[key]) for key in nf_chains.keys()}
        np.savez(name, **nf_chains)
        
        # Plot production
        name = outdir + f'results_production.npz'
        logger.info(f"Saving production results to {name}")
        state = get_sampler_state(jim, training=False)
        log_prob, local_accs, global_accs = state["log_prob"], state["local_accs"], state["global_accs"]
        local_accs = jnp.mean(local_accs, axis=0)
        global_accs = jnp.mean(global_accs, axis=0)
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)
        
        utils.plot_accs(local_accs, "Local accs (production)", "local_accs_production", outdir)
        utils.plot_accs(global_accs, "Global accs (production)", "global_accs_production", outdir)
        utils.plot_log_prob(log_prob, "Log probability (production)", "log_prob_production", outdir)
        
        # Finally, copy over this script to the outdir for reproducibility
        shutil.copy2(__file__, outdir + "copy_analysis.py")
        
        # Show the runtime
        end_time = time.time()
        runtime = end_time - start_time
        logger.info(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")
        with open(outdir + 'runtime.txt', 'w') as file:
            file.write(str(runtime))
        
        # Final cornerplot
        logger.info("Creating the final corner plot")

        try:
            # Get samples from jim - these are already transformed back to prior parameter space
            chains = jim.get_samples(training=False)
            chains = {key: np.array(chains[key]) for key in chains.keys()}
            #Check if a1 and a2 are present in the prior
            if (not pipe.complete_prior.has_param("a_1")) or (not pipe.complete_prior.has_param("a_2")):
                logger.info("adding chains for a1 and a2 based on tidal deformability")
                a1_chains, a2_chains = L1_L2_to_a1_a2(chains['lambda_1'], chains['lambda_2'])
                chains["a_1"] = np.asarray(a1_chains)
                chains["a_2"] = np.asarray(a2_chains)

            logger.info("Dumping the final production chains")
            np.savez(outdir + f'chains_production.npz', **chains)

            # Extract only the parameters we want to plot
            chains = np.array([chains[key].flatten() for key in pipe.keys_to_plot])
            logger.info(f"Chains shape is: {chains.shape}")

            utils.plot_chains(chains.T, "corner", outdir, labels=pipe.labels_to_plot, truths=truths)
        except Exception as e:
            logger.warning(f"Did not manage to create the cornerplot, exception was: {e}")
        
        logger.info("Finished successfully!")

    
    
    print("SAMPLER: ", pipe.config["sampler"])
    sampler =  pipe.config["sampler"]

    if sampler != "flowMC" and sampler != "blackjax-ns":
        logger.warning("Invalid sampler provided, defaulting to flowMC")
        sampler = 'flowMC'
    
    if sampler == "flowMC":
        logger.info("Using FlowMC as the sampler")
        run_flowMC()

    if sampler == "blackjax-ns":
        logger.info("Using BlackJax-NS as the sampler")
        # Setup transforms and functions for blackjax_ns
        print("SAMPLER: Setting up blackjax_ns sampler")
        #pipe.config['sample_transforms'] = setup_sample_transforms(pipe.complete_prior.base_prior, ifos=pipe.config["ifos"], phase_marginalization=False) #CHECK IF ALWAYS FALSE
        pipe.config["sample_transforms"] = pipe.sample_transforms
        pipe.config['likelihood_transforms'] = pipe.likelihood_transforms
        pipe.config['logprior_fn'] = create_logprior_fn(pipe.complete_prior, pipe.config['sample_transforms'])
        pipe.config['loglikelihood_fn'] = create_loglikelihood_fn(pipe.likelihood, pipe.config['sample_transforms'], pipe.config['likelihood_transforms'])
        pipe.config['unit_cube_stepper'] = create_unit_cube_stepper(pipe.complete_prior, pipe.config['sample_transforms'])

        print("SAMPLER: Running blackjax_ns sampler")
        samples_df = run_blackjax_ns_gw(pipe.config, pipe.complete_prior)

        # Save samples
        print("Saving blackjax_ns samples")
        samples_df.to_csv('blackjax_ns_samples.csv', index=False)

        # Add anything else you want to save from blackjax_ns here. The run_blackjax_ns_gw function will likely need to be modified to return more data.
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")
        
        print(f"Saving runtime")
        with open(outdir + 'runtime.txt', 'w') as file:
            file.write(str(runtime))

        # Fetch injected values for the plotting below
        if pipe.gw_pipe.is_gw_injection:
            logger.info("Fetching the injected values for plotting")
            with open(os.path.join(pipe.outdir, "injection.json"), "r") as f:
                injection = json.load(f)
            #Transfrom C1 and C2 to f_stop if needed
            if "C_1" and "C_2" in injection:
                print("Transforming C1 and C2 to f_stop for injection values")
                m1, m2  = Mc_q_to_m1_m2(injection["M_c"], injection["q"])
                f_stop = C1_C2_to_f_stop(injection["C_1"], injection["C_2"], m1, m2)
                injection["f_stop"] = f_stop
                del injection["C_1"]
                del injection["C_2"]
            #Add a1 and a2 from Lambda1 and Lambda2 if needed
            #TODO: add extra check if TF2_SSM is used
            #if "a_1" and "a_2" not in injection:
            if (not pipe.complete_prior.has_param("a_1")) or (not pipe.complete_prior.has_param("a_2")):
                print("Calculating a1 and a2 from tidal deformability")
                a1, a2 = L1_L2_to_a1_a2(injection["lambda_1"], injection['lambda_2'])
                injection["a_1"] = a1
                injection["a_2"] = a2
            print("KEYS TO PLOT: ", pipe.keys_to_plot)
            truths = np.array([injection[key] for key in pipe.keys_to_plot])        
        else:
            truths = None

        #Plotting the chains
        try:
            samples_df = pd.read_csv(outdir + 'blackjax_ns_samples.csv')
            logger.info("Loaded chains from CSV")
            chains = {col: samples_df[col].to_numpy() for col in samples_df.columns}
            #Check if a1 and a2 are present in the prior
            if (not pipe.complete_prior.has_param("a_1")) or (not pipe.complete_prior.has_param("a_2")):
                logger.info("adding chains for a1 and a2 based on tidal deformability")
                a1_chains, a2_chains = L1_L2_to_a1_a2(chains['lambda_1'], chains['lambda_2'])
                chains["a_1"] = np.asarray(a1_chains)
                chains["a_2"] = np.asarray(a2_chains)
            if (not pipe.complete_prior.has_param("f_stop")):
                m1, m2 = Mc_q_to_m1_m2(chains['M_c'], chains["q"])
                chains["f_stop"] = C1_C2_to_f_stop(chains["C_1"], chains["C_2"], m1, m2)

            logger.info("Dumping the final production chains")
            np.savez(outdir + f'chains_production.npz', **chains)

            # Extract only the parameters we want to plot
            chains = np.array([chains[key].flatten() for key in pipe.keys_to_plot])
            logger.info(f"Chains shape is: {chains.shape}")

            utils.plot_chains(chains.T, "corner", outdir, labels=pipe.labels_to_plot, truths=truths)
            logger.info("Cornerplot created succesfully")
        except Exception as e:
            logger.warning(f"Did not manage to create the cornerplot, exception was: {e}")
        
        
        print("Finished injection recovery successfully!")

############
### MAIN ###
############

def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python -m ninjax.analysis <outdir>")
    config_filename = sys.argv[1]
    pipe = NinjaxPipe(config_filename)
    if pipe.run_sampler:
        body(pipe)
    
if __name__ == "__main__":
    main()