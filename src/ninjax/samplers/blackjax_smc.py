"""BlackJAX Sequential Monte Carlo (SMC) with adaptive tempering for ninjax.

Adapted from jim-catalogue/scripts/analysis/jim/samplers/blackjax_smc.py
"""
import os
import json
import time
import jax
from jax.flatten_util import ravel_pytree
import jax.random
import jax.numpy as jnp
from blackjax import inner_kernel_tuning, adaptive_tempered_smc
from blackjax.smc import extend_params
from blackjax.smc.resampling import systematic
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax import rmh
from blackjax.mcmc import random_walk
import pandas as pd


class BlackJaxSMCSampler:
    """BlackJAX Sequential Monte Carlo with adaptive tempering."""

    def __init__(self, pipe):
        """Initialize sampler with ninjax pipeline.

        Parameters
        ----------
        pipe : NinjaxPipe
            Ninjax pipeline object containing configuration, prior, likelihood, and transforms
        """
        self.pipe = pipe
        self.outdir = pipe.outdir if pipe.outdir.endswith("/") else pipe.outdir + "/"

        # Extract hyperparameters from config
        self.n_particles = int(pipe.config["blackjax_smc_n_particles"])
        self.n_mcmc_steps = int(pipe.config["blackjax_smc_n_mcmc_steps"])
        self.target_ess = float(pipe.config["blackjax_smc_target_ess"])
        self.scale_proposal = float(pipe.config["blackjax_smc_scale_proposal"])

        print("=" * 70)
        print(f"BlackJAX Sequential Monte Carlo (Adaptive Tempering)")
        print("=" * 70)
        print(f"Configuration: {self.n_particles} particles, {self.n_mcmc_steps} MCMC steps per tempering stage")
        print(f"Target ESS: {self.target_ess}")
        print(f"MCMC Kernel: Gaussian Random Walk with adaptive covariance")

    def sample(self, rng_key):
        """Run SMC sampling with adaptive tempering.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            JAX random key

        Returns
        -------
        dict
            Dictionary with keys: "samples", "log_prob", "metadata"
        """
        # Create logprior_fn and loglikelihood_fn from pipe
        logprior_fn = self.pipe.create_logprior_fn()
        loglikelihood_fn = self.pipe.create_loglikelihood_fn()
        periodic_handler = self.pipe.create_periodic_handler()

        # Sample initial positions in prior space and transform to sampling space
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = self.pipe.complete_prior.sample(subkey, self.n_particles)
        for transform in self.pipe.sample_transforms:
            initial_position = jax.vmap(transform.forward)(initial_position)

        # Build random walk kernel
        kernel = random_walk.build_additive_step()

        def step(key, state, logdensity, cov):
            """Random walk Metropolis-Hastings step with Gaussian proposal and periodic wrapping."""
            def proposal_distribution(key, position):
                x, ravel_fn = ravel_pytree(position)
                raw_proposal = ravel_fn(jax.random.multivariate_normal(key, jnp.zeros_like(x), cov))
                # Apply periodic wrapping using the wrapper from config
                proposal = periodic_handler(raw_proposal, position)
                return proposal

            return kernel(
                key,
                state,
                logdensity,
                proposal_distribution,
            )

        # Compute initial covariance matrix from particles
        cov = particles_covariance_matrix(initial_position)
        init_params = {"cov": cov}

        def mcmc_parameter_update_fn(key, state, info):
            """Update covariance matrix based on current particles."""
            cov = particles_covariance_matrix(state.particles)
            return extend_params({"cov": cov * self.scale_proposal})

        # Initialize SMC algorithm
        smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=step,
            mcmc_init_fn=rmh.init,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params(init_params),
            target_ess=self.target_ess,
            num_mcmc_steps=self.n_mcmc_steps,
        )

        state = smc_alg.init(initial_position)  # type: ignore

        def cond_fn(carry):
            state, _, _, _, _, _, _ = carry
            # state is StateWithParameterOverride, state[0] is the first element (sampler_state)
            # state.sampler_state is TemperedSMCState
            return state.sampler_state.lmbda < 1

        def body_fn(carry):
            state, key, step_count, lmbda_history, ess_history, acceptance_history, logz_increments = carry
            key, subkey = jax.random.split(key, 2)

            state, info = smc_alg.step(subkey, state)

            # Compute ESS for this step
            weights = state.sampler_state.weights  # type: ignore
            ess_value = jnp.sum(weights) ** 2 / jnp.sum(weights**2) / self.n_particles

            # Extract acceptance rate from info
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore

            # Extract log evidence increment
            log_evidence_increment = info.log_likelihood_increment  # type: ignore

            # Update histories
            lmbda_history = lmbda_history.at[step_count].set(state.sampler_state.lmbda)  # type: ignore
            ess_history = ess_history.at[step_count].set(ess_value)
            acceptance_history = acceptance_history.at[step_count].set(acceptance_rate)
            logz_increments = logz_increments.at[step_count].set(log_evidence_increment)

            return (state, key, step_count + 1, lmbda_history, ess_history, acceptance_history, logz_increments)

        # Run SMC with JAX while_loop
        print("\nStarting SMC sampling with adaptive tempering...")
        sampling_start_time = time.time()

        # Pre-allocate arrays for tracking (max 1000 steps should be more than enough)
        max_steps = 1000
        lmbda_history = jnp.zeros(max_steps)
        ess_history = jnp.zeros(max_steps)
        acceptance_history = jnp.zeros(max_steps)
        logz_increments = jnp.zeros(max_steps)

        init_carry = (state, rng_key, 0, lmbda_history, ess_history, acceptance_history, logz_increments)
        state, rng_key, steps, lmbda_history, ess_history, acceptance_history, logz_increments = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        steps = int(steps)

        # Trim histories to actual number of steps
        lmbda_history = lmbda_history[:steps]
        ess_history = ess_history[:steps]
        acceptance_history = acceptance_history[:steps]
        logz_increments = logz_increments[:steps]

        # Compute log evidence
        logZ = float(jnp.sum(logz_increments))

        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time
        print(f"\nSMC sampling completed! Performed {steps} annealing steps.")
        print(f"Sampling time: {(sampling_time)//60:.0f} minutes {(sampling_time)%60:.1f} seconds")

        # Extract final particles and weights
        final_particles = state.sampler_state.particles  # type: ignore
        final_weights = state.sampler_state.weights  # type: ignore

        # Compute ESS
        ess = jnp.sum(final_weights) ** 2 / jnp.sum(final_weights**2)

        print(f"\nEvidence: log(Z) = {logZ:.2f}")
        print(f"Effective sample size: {ess:.1f} ({ess/self.n_particles*100:.1f}%)")

        # Transform particles back to prior space
        print("\nTransforming samples back to prior space...")
        physical_particles = final_particles
        for transform in reversed(self.pipe.sample_transforms):
            physical_particles = jax.vmap(transform.backward)(physical_particles)

        # Compute log-likelihoods for final particles
        print("Computing log-likelihoods for final particles...")

        samples_logL = physical_particles.copy()
        for likelihood_transform in self.pipe.likelihood_transforms:
            samples_logL = jax.vmap(likelihood_transform.forward)(samples_logL)

        logL = jax.vmap(self.pipe.likelihood.evaluate)(samples_logL, {})

        # Create DataFrame with samples and metadata
        samples_dict = physical_particles.copy()
        samples_dict['logL'] = logL
        samples_dict['weight'] = final_weights

        df = pd.DataFrame(samples_dict)

        # Prepare metadata
        metadata = {
            'sampler': 'blackjax-smc',
            'seed': int(self.pipe.sampling_seed) if self.pipe.sampling_seed is not None else None,
            # Sampler configuration
            'n_particles': self.n_particles,
            'n_mcmc_steps': self.n_mcmc_steps,
            'target_ess': float(self.target_ess),
            'scale_proposal': self.scale_proposal,
            # Sampling results
            'sampling_time_minutes': sampling_time / 60,
            'n_samples': len(df),
            'n_annealing_steps': steps,
            'logZ': logZ,
            'effective_sample_size': float(ess),
            'n_likelihood_evaluations': self.n_mcmc_steps * steps * self.n_particles,
            # Diagnostic histories (for later plotting)
            'lmbda_history': lmbda_history.tolist(),
            'ess_history': ess_history.tolist(),
            'acceptance_history': acceptance_history.tolist(),
        }

        # Return unified result dictionary
        return {
            "samples": df,  # pandas DataFrame with weights
            "log_prob": logL,
            "metadata": metadata
        }
