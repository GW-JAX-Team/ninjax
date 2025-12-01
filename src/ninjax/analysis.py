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

from jimgw.core.jim import Jim

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger
from ninjax.pipes.ninjax_pipe import NinjaxPipe

####################
### Script setup ###
####################

def body(pipe: NinjaxPipe):
    start_time = time.time()
    
    # Before main code, check if outdir is correct dir format
    outdir = pipe.outdir
    if outdir[-1] != "/":
        outdir += "/"
    logger.info(f"Saving output to {outdir}")
    
    jim_hyperparameters = pipe.jim_hyperparameters
    analysis_config = pipe.analysis_config

    # NOTE: The old API used a mass_matrix-based local_sampler_arg, but the new API
    # uses a scalar mala_step_size parameter. The mala_step_size is already set
    # in set_flowmc_hyperparameters() from the config's eps_mass_matrix value.

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
        truths = np.array([injection[key] for key in pipe.keys_to_plot])
    else:
        truths = None

    ### Finally, do the sampling
    jim.sample()
    jim.print_summary()

    # Plot training
    name = outdir + f'results_training.npz'
    logger.info(f"Saving samples to {name}")
    state = jim.Sampler.get_sampler_state(training = True)
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
    logger.info("Saving the NF")
    jim.Sampler.save_flow(outdir + "nf_model")
    name = outdir + 'results_NF.npz'
    nf_chains = jim.Sampler.sample_flow(10_000)
    np.savez(name, chains = nf_chains)
    
    # Plot production
    name = outdir + f'results_production.npz'
    state = jim.Sampler.get_sampler_state(training = False)
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
        chains = jim.get_samples(training = False)
        chains = pipe.likelihood.transform(chains)
        chains = {key: np.array(chains[key]) for key in chains.keys()}
        
        logger.info("Dumping the final production chains")
        np.savez(outdir + f'chains_production.npz', **chains)
        
        chains = np.array([chains[key].flatten() for key in pipe.keys_to_plot])
        logger.info(f"Chains shape is: {chains.shape}")
        
        utils.plot_chains(chains.T, "corner", outdir, labels = pipe.labels_to_plot, truths = truths)
    except Exception as e:
        logger.warning(f"Did not manage to create the cornerplot, exception was: {e}")
    
    logger.info("Finished successfully!")

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