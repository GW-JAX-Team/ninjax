"""BlackJAX Nested Slice Sampling (NSS) for ninjax.

Adapted from jim-catalogue/scripts/analysis/jim/samplers/blackjax_nss.py

NSS combines nested sampling with an adaptive slice sampling inner kernel.
Unlike the acceptance walk kernel, NSS doesn't require unit cube transforms
- it works directly in the prior space (or with BoundToUnbound transforms).
"""
import os
import json
import time
import jax
import jax.random
import jax.numpy as jnp
import tqdm
import blackjax
from blackjax.ns.utils import finalise, ess as ns_ess
from anesthetic.samples import NestedSamples


class BlackJaxNSSSampler:
    """BlackJAX Nested Slice Sampling."""

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
        self.n_live = int(pipe.config["blackjax_nss_n_live"])
        self.n_delete_frac = float(pipe.config["blackjax_nss_n_delete_frac"])
        self.num_inner_steps = int(pipe.config["blackjax_nss_num_inner_steps"])
        self.termination_dlogz = float(pipe.config["blackjax_nss_termination_dlogz"])
        self.n_delete = int(self.n_live * self.n_delete_frac)

        # Calculate n_dims and actual num_mcmc_steps
        self.n_dims = len(pipe.complete_prior.parameter_names)
        self.num_mcmc_steps = self.num_inner_steps * self.n_dims

        print("=" * 70)
        print(f"BlackJAX Nested Slice Sampling (NSS)")
        print("=" * 70)
        print(f"Configuration: {self.n_live} live points, batch size {self.n_delete}")
        print(f"Inner MCMC steps: {self.num_mcmc_steps} ({self.num_inner_steps} × {self.n_dims} dimensions)")
        print(f"Termination condition: dlogZ < {self.termination_dlogz}")

    def sample(self, rng_key):
        """Run nested slice sampling.

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
        initial_position = self.pipe.complete_prior.sample(subkey, self.n_live)
        for transform in self.pipe.sample_transforms:
            initial_position = jax.vmap(transform.forward)(initial_position)

        # Initialize NSS sampler with stepper from config
        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=self.n_delete,
            num_inner_steps=self.num_mcmc_steps,
            stepper_fn=periodic_handler,
        )

        state = nested_sampler.init(initial_position)

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = nested_sampler.step(subk, state)
            return (state, k), dead_point

        def terminate(state):
            """Termination condition: stop when remaining evidence contribution is small.

            Uses the same condition as NS-AW: dlogz = log(1 + exp(logZ_live - logZ))
            This represents the log of (1 + remaining fractional evidence).
            """
            dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
            return jnp.isfinite(dlogz) and dlogz < self.termination_dlogz

        # Run nested sampling
        print("\nStarting nested sampling with slice sampling kernel...")
        sampling_start_time = time.time()
        dead = []

        with tqdm.tqdm(desc="NSS Dead points", unit=" dead points") as pbar:
            while not terminate(state):
                (state, rng_key), dead_info = one_step((state, rng_key), None)
                dead.append(dead_info)
                pbar.update(self.n_delete)

        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time
        print(f"\nNested sampling completed! Generated {len(dead) * self.n_delete} dead points.")
        print(f"Sampling time: {(sampling_time)//60:.0f} minutes {(sampling_time)%60:.1f} seconds")

        # Finalize results - combines dead points with final live points
        print("\nFinalizing nested sampling results...")
        final_info = finalise(state, dead)

        # Compute ESS using BlackJAX (Kish ESS, beta=2)
        rng_key, ess_key = jax.random.split(rng_key)
        kish_ess = float(ns_ess(ess_key, final_info))
        print(f"Kish ESS = {kish_ess:.1f}")

        # Extract slice sampling statistics and compute likelihood evaluations
        # Each slice step calls the constraint function (likelihood) during:
        # - Stepping-out phase: num_steps evaluations to expand the interval
        # - Shrinking phase: num_shrink evaluations to find valid point
        total_num_steps = int(jnp.sum(final_info.inner_kernel_info.info.num_steps))  # type: ignore
        total_num_shrink = int(jnp.sum(final_info.inner_kernel_info.info.num_shrink))  # type: ignore
        n_likelihood_evaluations = total_num_steps + total_num_shrink
        print(f"Likelihood evaluations: {n_likelihood_evaluations:,} (stepping-out: {total_num_steps:,}, shrinking: {total_num_shrink:,})")

        # Transform all particles back to prior space
        print("\nTransforming samples back to prior space...")
        physical_particles = final_info.particles
        for transform in reversed(self.pipe.sample_transforms):
            physical_particles = jax.vmap(transform.backward)(physical_particles)

        # Handle logL_birth NaN values
        logL_birth = final_info.loglikelihood_birth.copy()
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        # Create NestedSamples object for proper nested sampling analysis
        df = NestedSamples(
            physical_particles,
            logL=final_info.loglikelihood,
            logL_birth=logL_birth,
            logzero=jnp.nan,
            dtype=jnp.float64,
        )

        # Compute log evidence from anesthetic with bootstrap error
        logZ_samples = df.logZ(nsamples=1000)
        logZ = float(logZ_samples.mean())
        logZ_err = float(logZ_samples.std())
        print(f"\nEvidence: log(Z) = {logZ:.2f} ± {logZ_err:.2f}")

        # Extract posterior samples
        samples = df.posterior_points()

        # Prepare metadata
        metadata = {
            'sampler': 'blackjax-nss',
            'seed': int(self.pipe.sampling_seed) if self.pipe.sampling_seed is not None else None,
            # Sampler configuration
            'n_live': self.n_live,
            'n_delete': self.n_delete,
            'n_delete_frac': self.n_delete_frac,
            'num_inner_steps': self.num_inner_steps,
            'num_mcmc_steps': self.num_mcmc_steps,
            'termination_dlogz': self.termination_dlogz,
            # Sampling results
            'sampling_time_minutes': sampling_time / 60,
            'n_samples': len(samples),
            'n_dead_points': len(dead) * self.n_delete,
            'logZ': logZ,
            'logZ_err': logZ_err,
            'kish_ess': kish_ess,
            'entropy_ess': df.neff(),
            # Likelihood evaluation statistics
            'n_likelihood_evaluations': n_likelihood_evaluations,
            'slice_stepping_out_evals': total_num_steps,
            'slice_shrinking_evals': total_num_shrink,
        }

        # Return unified result dictionary
        return {
            "samples": samples,  # pandas DataFrame
            "log_prob": final_info.loglikelihood,
            "metadata": metadata
        }
