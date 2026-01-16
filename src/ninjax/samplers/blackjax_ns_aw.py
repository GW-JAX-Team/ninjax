"""BlackJAX nested sampling with acceptance walk kernel for ninjax.

Adapted from jim-catalogue/scripts/analysis/jim/samplers/blackjax_ns_aw.py
"""
import os
import json
import time
import jax
import jax.random
import jax.numpy as jnp
import tqdm
from blackjax.ns.utils import finalise, ess as ns_ess
from anesthetic.samples import NestedSamples

from ninjax.samplers.acceptance_walk_kernel import bilby_adaptive_de_sampler_unit_cube as acceptance_walk_sampler


class BlackJaxNSAWSampler:
    """BlackJAX Nested Sampling with Acceptance Walk kernel."""

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
        self.n_live = int(pipe.config["blackjax_ns_aw_n_live"])
        self.n_delete_frac = float(pipe.config["blackjax_ns_aw_n_delete_frac"])
        self.n_target = int(pipe.config["blackjax_ns_aw_n_target"])
        self.max_mcmc = int(pipe.config["blackjax_ns_aw_max_mcmc"])
        self.max_proposals = int(pipe.config["blackjax_ns_aw_max_proposals"])
        self.termination_dlogz = float(pipe.config["blackjax_ns_aw_termination_dlogz"])
        self.n_delete = int(self.n_live * self.n_delete_frac)

        print("=" * 70)
        print(f"BlackJAX Nested Sampling (Acceptance Walk)")
        print("=" * 70)
        print(f"Configuration: {self.n_live} live points, batch size {self.n_delete}")
        print(f"Termination condition: dlogZ < {self.termination_dlogz}")

    def sample(self, rng_key):
        """Run nested sampling.

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

        # Initialize nested sampler
        nested_sampler = acceptance_walk_sampler(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            nlive=self.n_live,
            n_target=self.n_target,
            max_mcmc=self.max_mcmc,
            num_delete=self.n_delete,
            stepper_fn=periodic_handler,
            max_proposals=self.max_proposals
        )

        state = nested_sampler.init(initial_position)

        def terminate(state):
            """Termination condition: stop when remaining evidence is small."""
            dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
            return jnp.isfinite(dlogz) and dlogz < self.termination_dlogz

        step_fn = jax.jit(nested_sampler.step)

        # Run nested sampling
        print("Starting nested sampling...")
        sampling_start_time = time.time()
        dead = []
        with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
            while not terminate(state):
                rng_key, subkey = jax.random.split(rng_key, 2)
                state, dead_info = step_fn(subkey, state)
                dead.append(dead_info)
                pbar.update(self.n_delete)

        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time
        print(f"\nNested sampling completed! Generated {len(dead) * self.n_delete} dead points.")
        print(f"Sampling time: {(sampling_time)//60:.0f} minutes {(sampling_time)%60:.1f} seconds")

        # Finalize results
        print("\nFinalizing nested sampling results...")
        final_state = finalise(state, dead)

        # Compute ESS using BlackJAX (Kish ESS, beta=2)
        rng_key, ess_key = jax.random.split(rng_key)
        kish_ess = float(ns_ess(ess_key, final_state))
        print(f"Kish ESS = {kish_ess:.1f}")

        # Transform all particles back to prior space
        print("\nTransforming samples back to prior space...")
        physical_particles = final_state.particles
        for transform in reversed(self.pipe.sample_transforms):
            physical_particles = jax.vmap(transform.backward)(physical_particles)

        logL_birth = final_state.loglikelihood_birth.copy()
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        df = NestedSamples(
            physical_particles,
            logL=final_state.loglikelihood,
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
            'sampler': 'blackjax-ns-aw',
            'seed': int(self.pipe.sampling_seed) if self.pipe.sampling_seed is not None else None,
            # Sampler configuration
            'n_live': self.n_live,
            'n_delete': self.n_delete,
            'n_delete_frac': self.n_delete_frac,
            'n_target': self.n_target,
            'max_mcmc': self.max_mcmc,
            'max_proposals': self.max_proposals,
            'termination_dlogz': self.termination_dlogz,
            # Sampling results
            'sampling_time_minutes': sampling_time / 60,
            'n_samples': len(samples),
            'logZ': logZ,
            'logZ_err': logZ_err,
            'kish_ess': kish_ess,
            'entropy_ess': df.neff(),
            'n_likelihood_evaluations': int(sum(final_state.inner_kernel_info.n_likelihood_evals))
        }

        # Return unified result dictionary
        return {
            "samples": samples,  # pandas DataFrame
            "log_prob": final_state.loglikelihood,
            "metadata": metadata
        }
