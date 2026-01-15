# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ninjax` (Nitro-speed inference with JAX) is a Python package for injection-recovery tests in gravitational wave (GW) inference. It integrates with:
- `flowMC`: JAX-based Bayesian inference with normalizing flows
- `jim`: JAX-based gravitational wave inference
- `ripple`: JAX-based waveform models

The package is designed for fast, GPU-accelerated Bayesian parameter estimation of GW signals using modern flow-based sampling methods.

## Development Commands

### Installation
```bash
# Standard install
pip install -e .

# Install with uv (if available)
uv pip install -e .
```

### Running Analysis
```bash
# Main entry point (requires config.ini and prior.prior in outdir)
ninjax_analysis <outdir>

# Example runs
cd examples/example_1
ninjax_analysis .
```

### Testing
```bash
# Run single test
python test/test_deprecated_keys.py

# Type checking (uses pyright as configured in pyproject.toml)
pyright src/ninjax/
```

### Running a Single Test
To test individual components without running the full pipeline:
```bash
# Test deprecated key validation
python test/test_deprecated_keys.py
```

## Architecture

### Core Pipeline Flow
1. **Config Loading** (`ninjax_pipe.py`): Loads `config.ini`, merges with defaults from `default_config.ini`
2. **Prior Setup** (`ninjax_pipe.py`): Parses `prior.prior` file to build `CombinePrior` object
3. **GW-Specific Setup** (`gw_pipe.py`): Sets up waveforms, detectors, PSDs, and injections/data
4. **Likelihood Construction** (`ninjax_pipe.py`): Builds jim likelihood with transforms
5. **Sampler Setup** (`ninjax_pipe.py`): Configures flowMC hyperparameters
6. **Sampling** (`analysis.py`): Runs training and production sampling loops
7. **Output** (`analysis.py`): Saves chains, plots, and diagnostics

### Key Classes

**`NinjaxPipe`** (`src/ninjax/pipes/ninjax_pipe.py`)
- Main orchestrator class
- Handles config loading, prior parsing, transform setup, and sampler configuration
- Delegates GW-specific tasks to `GWPipe`

**`GWPipe`** (`src/ninjax/pipes/gw_pipe.py`)
- GW-specific pipeline logic
- Sets up waveforms, detectors, PSDs
- Handles injection generation with SNR thresholds
- Supports overlapping signals (untested)

**Main Script** (`src/ninjax/analysis.py`)
- Entry point via `ninjax_analysis` command
- Runs full sampling pipeline
- Generates plots and saves results

### Configuration System

**Config File Format** (`config.ini`):
- INI format with defaults in `src/ninjax/pipes/default_config.ini`
- Must be present in outdir alongside `prior.prior`
- Key sections: likelihood type, GW parameters, flowMC hyperparameters

**Prior File Format** (`prior.prior`):
- Python-like syntax evaluated at runtime
- Each line defines a prior: `<name>_prior = <PriorClass>(..., parameter_names=['<param>'])`
- Example: `Mc_prior = UniformPrior(1.0, 2.2, parameter_names=['M_c'])`
- Supports: `UniformPrior`, `PowerLawPrior`, `SinePrior`, `CosinePrior`, `NFPrior` (custom)

### Detector Support
- **Tested**: H1, L1, V1 (LIGO Hanford, Livingston, Virgo)
- **Available (untested)**: ET (Einstein Telescope - returns 3 detectors: ET1, ET2, ET3), CE (Cosmic Explorer)

### Waveform Models
- `TaylorF2`: Post-Newtonian BNS waveform
- `IMRPhenomD_NRTidalv2`: IMR waveform with tidal effects for BNS
- `IMRPhenomD`: IMR waveform for BBH (no tidal)
- `IMRPhenomPv2`: Precessing BBH (untested)

### Likelihood Types
- `HeterodynedTransientLikelihoodFD`: Heterodyned likelihood (tested, recommended)
- `BaseTransientLikelihoodFD`: Non-heterodyned likelihood (untested)

## Critical Known Issues

### High Priority Technical Debt

1. **NFPrior Hardcoding** (`src/ninjax/prior.py`):
   - Hardcoded to 4D BNS priors: `shape = (40_000, 4)`, `nn_depth=5`, `nn_block_dim=8`
   - Assumes parameter order: `[m_1, m_2, lambda_1, lambda_2]`
   - **Fix needed**: Infer dimensions from `len(parameter_names)`, make architecture configurable

2. **Deprecated Code to Remove**:
   - `LikelihoodWithTransforms` in `likelihood.py` (no longer used with new jim API)
   - Most transform functions in `transforms.py` (90% deprecated by jim's Transform API)
   - Evaluate `binary_Love`, `m1_m2_to_Mc_q`, `spin_sphere_to_cartesian_*` before removing

3. **Hardcoded Transform Pipeline** (`ninjax_pipe.py:set_likelihood_transforms`):
   - Always returns `[MassRatioToSymmetricMassRatioTransform]`
   - Should be config-driven to support different transform needs

4. **Waveform Parameter Extraction** (`gw_pipe.py`):
   - Old jim API had `required_keys` attribute (now removed)
   - Current workaround uses all injection keys
   - **Improvement**: Use function signature inspection to determine required parameters

### Untested Features (May Be Broken)

1. **BaseTransientLikelihoodFD** (non-heterodyned likelihood)
2. **Overlapping signal injections** (`set_overlapping_gw_injection`) - has massive code duplication
3. **Real data loading** (`set_gw_data_from_npz`)
4. **Precessing waveforms** (IMRPhenomPv2)
5. **EOS-based lambda injection** (`inject_lambdas_from_eos`)
6. **ET and CE detectors** (recently added support, not yet tested)

### Configuration Key Migration

**Deprecated keys** (will raise `ValueError` with helpful messages):
- `n_loop_training` → `n_training_loops`
- `n_loop_production` → `n_production_loops`
- `max_samples` → `n_max_examples`
- `train_thinning` → `local_thinning`
- `output_thinning` → `global_thinning`
- `eps_mass_matrix` → `mala_step_size`
- `num_layers` → `rq_spline_n_layers`
- `hidden_size` → `rq_spline_hidden_units`
- `num_bins` → `rq_spline_n_bins`

Removed parameters (no replacement): `momentum`, `use_global`, `keep_quantile`, `n_sample_max`, `local_sampler_arg`, `nf_model_kwargs`

## Important Development Notes

### When Adding New Features

1. **Waveform Models**: Add to `WAVEFORMS_DICT` in `gw_pipe.py`, update `BNS_WAVEFORMS` list if applicable
2. **Likelihood Types**: Add to `LIKELIHOODS_DICT` in `ninjax_pipe.py`
3. **Transform Types**: Use jim's Transform API (avoid adding to deprecated `transforms.py`)
4. **Prior Types**: Add custom priors to `prior.py` (avoid modifying jim's priors)

### Output Files

Each run produces in `outdir`:
- `complete_config.json`: Full configuration used
- `injection.json`: True parameter values (if injection)
- `*.npz` files: Chains, log probabilities, acceptance rates
- `*.png` files: Diagnostic and corner plots
- `nf_model.eqx`: Trained normalizing flow model (equinox format)
- `log.out`: Detailed execution log
- `runtime.txt`: Total runtime

### File Organization

```
src/ninjax/
├── analysis.py              # Main entry point
├── pipes/
│   ├── ninjax_pipe.py      # Main orchestration pipeline
│   ├── gw_pipe.py          # GW-specific setup
│   ├── pipe_utils.py       # Utility functions (large, needs refactoring)
│   └── default_config.ini  # Default configuration values
├── parser.py               # Config file parsing
├── prior.py                # Custom prior definitions (NFPrior)
├── likelihood.py           # Custom likelihood classes (mostly deprecated)
├── transforms.py           # Transform functions (mostly deprecated)
├── AdV_psd.txt            # Advanced Virgo PSD
└── AplusDesign_PSD.txt    # A+ design LIGO PSD
```

### Common Pitfalls

1. **PSD file paths**: Default config uses absolute paths - update for local machine
2. **Prior file syntax**: Must use exact jim prior class names (case-sensitive)
3. **Parameter names**: Must match between prior file, config, and waveform requirements
4. **Chirp mass recentering**: If `center_chirp_mass_prior = True`, prior will be modified post-injection
5. **SNR thresholds**: Injection generation retries until SNR is in range - can be slow if range is narrow

### jim API Migration Status

Successfully migrated to new jim API (December 2025). Key changes:
- Transforms now use jim's `Transform` classes instead of custom implementations
- Likelihood setup uses direct jim classes instead of wrappers
- `required_keys` attribute removed from waveforms
- flowMC hyperparameters reorganized (see deprecated keys list)

See `NEXT_STEPS.md` for comprehensive technical debt, testing priorities, and improvement roadmap.
