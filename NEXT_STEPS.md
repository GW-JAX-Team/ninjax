# NEXT STEPS for ninjax Development

**Date**: 2025-12-06
**Context**: First successful end-to-end test with new jim/flowMC API completed (example_1)

This document outlines critical next steps for improving ninjax code quality, testing coverage, and maintainability after the successful migration to the new jim API.

---

## 1. CRITICAL: Remove Technical Debt (Priority: HIGH)

### 1.1 Deprecated Code Cleanup

**Files to clean up:**

1. **`src/ninjax/transforms.py`** - MOSTLY DEPRECATED
   - **Status**: 90% of functions are deprecated by new jim API
   - **Action**:
     - Remove deprecated functions: `q_to_eta`, `cos_iota_to_iota`, `sin_dec_to_dec`
     - Evaluate `binary_Love` - port to new Transform API if useful
     - Evaluate `m1_m2_to_Mc_q`, `spin_sphere_to_cartesian_*` - keep only if needed
     - Add deprecation warnings to remaining functions
   - **Risk**: Low (none of these are used in example_1)

2. **`src/ninjax/likelihood.py`** - Contains deprecated wrapper
   - **Status**: `LikelihoodWithTransforms` class is NO LONGER USED
   - **Action**:
     - DELETE `LikelihoodWithTransforms` class entirely
     - Test if `ZeroLikelihood` is used anywhere (grep codebase)
     - If `ZeroLikelihood` is unused, remove it
     - Consider implementing `CombinedLikelihood` if multi-event analysis is planned
   - **Risk**: Low (LikelihoodWithTransforms confirmed unused)

### 1.2 Fix Hardcoded Assumptions

**High priority fixes:**

1. **`src/ninjax/prior.py` - NFPrior class**
   - **Problem**: Hardcoded for 4D BNS priors only
     ```python
     shape = (40_000, 4)  # HARDCODED!
     nn_depth=5, nn_block_dim=8  # HARDCODED!
     # Assumes parameter order: [m_1, m_2, lambda_1, lambda_2]
     ```
   - **Action**:
     - Infer `shape[1]` from `len(parameter_names)`
     - Make `shape[0]`, `nn_depth`, `nn_block_dim` configurable via kwargs
     - Generalize constraint handling (currently assumes mass/lambda ordering)
     - Add validation that loaded NF matches expected dimensions
   - **Risk**: HIGH if anyone tries to use NFPrior with different dimensions

2. **`src/ninjax/pipes/ninjax_pipe.py` - Transform pipelines**
   - **Problem**: Transforms are hardcoded, not inferred from config/waveform
     ```python
     def set_likelihood_transforms(self):
         return [MassRatioToSymmetricMassRatioTransform]  # ALWAYS returns this!
     ```
   - **Action**:
     - Make transform pipelines configurable from config file
     - Auto-detect required transforms based on:
       - Prior parameter names (e.g., if "q" in priors, add q→eta transform)
       - Waveform model requirements
     - Support multiple transform types (mass, spin, angle)
   - **Risk**: MEDIUM - current hardcoding works but limits flexibility

3. **`src/ninjax/pipes/gw_pipe.py` - Waveform parameter extraction**
   - **Problem**: `required_keys` attribute removed in new jim API
     ```python
     # TODO: required_keys attribute removed in new jim API
     true_param = {key: float(injection[key]) for key in injection.keys()}
     ```
   - **Action**:
     - Implement proper parameter extraction based on waveform model
     - Use waveform model's signature or documentation to determine required params
     - Add validation that all required parameters are present
   - **Risk**: MEDIUM - currently works but fragile

---

## 2. TESTING: Validate Untested Features (Priority: HIGH)

### 2.1 Untested GW Features

**Features that have NOT been tested with new jim API:**

1. **BaseTransientLikelihoodFD (non-heterodyned likelihood)**
   - **File**: `src/ninjax/pipes/ninjax_pipe.py:439`
   - **Test**: Create `examples/example_base_likelihood/` with non-heterodyned setup
   - **Why important**: Heterodyned likelihood may not always be appropriate

2. **Overlapping signal injections**
   - **File**: `src/ninjax/pipes/gw_pipe.py:469` (`set_overlapping_gw_injection`)
   - **Test**: Create `examples/example_overlapping/` with two BNS signals
   - **Known issues**:
     - Massive code duplication with `set_gw_injection`
     - `inject_signal()` and `add_signal()` API may have changed
     - EOS injection for overlapping BNS raises NotImplementedError
   - **Why important**: Overlapping signals are scientifically interesting

3. **Real data loading (not injection)**
   - **File**: `src/ninjax/pipes/gw_pipe.py:618` (`set_gw_data_from_npz`)
   - **Test**: Create example with real LIGO data from GW150914 or similar
   - **Why important**: Main use case is analyzing real events, not just injections

4. **Precessing waveforms (IMRPhenomPv2)**
   - **File**: `src/ninjax/pipes/gw_pipe.py:19`
   - **Test**: Create `examples/example_precessing/` with spin priors
   - **Why important**: Many astrophysical systems have precessing spins

5. **EOS-based lambda injection**
   - **File**: `src/ninjax/pipes/pipe_utils.py:280` (`inject_lambdas_from_eos`)
   - **Test**: Create example with `eos_file` specified in config
   - **Why important**: Realistic BNS sampling should use EOS constraints

### 2.2 Testing Strategy

**Recommended test structure:**

```
ninjax/test/
├── unit/
│   ├── test_transforms.py         # Test individual transform functions
│   ├── test_prior.py               # Test NFPrior, other custom priors
│   ├── test_parser.py              # Test config parsing edge cases
│   └── test_utils.py               # Test utility functions (SNR, injection gen, etc.)
├── integration/
│   ├── test_base_likelihood.py     # Test non-heterodyned likelihood end-to-end
│   ├── test_overlapping.py         # Test overlapping injections
│   ├── test_real_data.py           # Test real data loading
│   └── test_precessing.py          # Test precessing waveforms
└── conftest.py                     # Shared fixtures
```

**Priority order:**
1. Unit tests for `NFPrior` (HIGH - currently very fragile)
2. Integration test for `BaseTransientLikelihoodFD`
3. Integration test for overlapping injections
4. Unit tests for `pipe_utils.py` functions
5. Integration test for real data loading

---

## 3. CODE QUALITY: Improve Modularity and Robustness (Priority: MEDIUM)

### 3.1 Automatic Waveform Parameter Handling

**Current problem**: Manual parameter extraction is error-prone

**Proposed solution**: Use function signature inspection

```python
# In ninjax_pipe.py or gw_pipe.py

import inspect
from jimgw.core.single_event.waveform import Waveform

def get_waveform_required_params(waveform: Waveform) -> list[str]:
    """Extract required parameters from waveform model signature

    This inspects the waveform's __call__ method to determine
    what parameters it expects, eliminating hardcoded assumptions.
    """
    # Get the signature of the waveform's __call__ method
    sig = inspect.signature(waveform.__call__)

    # Extract parameter names (excluding 'self' and 'frequencies')
    params = [
        name for name, param in sig.parameters.items()
        if name not in ['self', 'frequencies', 'f']
    ]

    return params

def validate_injection_params(injection: dict, waveform: Waveform):
    """Validate that injection contains all required waveform parameters"""
    required = get_waveform_required_params(waveform)
    missing = set(required) - set(injection.keys())

    if missing:
        raise ValueError(
            f"Injection missing required waveform parameters: {missing}\n"
            f"Required: {required}\n"
            f"Provided: {list(injection.keys())}"
        )
```

**Benefits**:
- No more hardcoded parameter lists
- Automatic validation
- Clear error messages when parameters are missing
- Works with any waveform model

**Implementation**:
- Add to `src/ninjax/pipes/gw_pipe.py`
- Use in `set_gw_injection()` before calling waveform
- Add unit tests

### 3.2 Config-Driven Transform Setup

**Current problem**: Transforms are hardcoded in `ninjax_pipe.py`

**Proposed solution**: Specify transforms in config file

**Example config.ini:**
```ini
# Transforms to apply (comma-separated)
# Available: q_to_eta, mass_ratio, periodic_phase, periodic_ra, periodic_psi
likelihood_transforms = mass_ratio
sample_transforms = periodic_phase, periodic_ra, periodic_psi
```

**Implementation:**
```python
# In ninjax_pipe.py

from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    # PeriodicTransform,  # TODO: check if available in jim
)

TRANSFORM_REGISTRY = {
    "mass_ratio": MassRatioToSymmetricMassRatioTransform,
    # "periodic_phase": lambda: PeriodicTransform(parameter_names=["phase_c"]),
    # Add more as needed
}

def set_likelihood_transforms(self) -> list:
    """Build likelihood_transforms from config file"""
    transform_str = self.config.get("likelihood_transforms", "mass_ratio")

    if transform_str.lower() == "none":
        return []

    transform_names = [t.strip() for t in transform_str.split(",")]
    transforms = []

    for name in transform_names:
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform '{name}'. "
                f"Available: {list(TRANSFORM_REGISTRY.keys())}"
            )
        transforms.append(TRANSFORM_REGISTRY[name])

    logger.info(f"Loaded transforms: {transform_names}")
    return transforms
```

**Benefits**:
- Users can customize transforms without modifying code
- Easy to add/remove transforms for experiments
- Consistent with other config-driven setup

### 3.3 Split Large Utility File

**Current problem**: `pipe_utils.py` contains many unrelated functions

**Proposed structure:**
```
src/ninjax/utils/
├── __init__.py              # Re-export for backward compatibility
├── plotting.py              # plot_chains, plot_accs, plot_log_prob, corner utils
├── io_utils.py              # directory checks, CustomJSONEncoder, logger setup
├── physics.py               # signal_duration, compute_snr
└── injection.py             # generate_injection, inject_lambdas_from_eos
```

**Benefits**:
- Easier to navigate
- Clearer organization
- Easier to test individual modules
- Backward compatible via `__init__.py` re-exports

---

## 4. ROBUSTNESS: Improve Error Handling and Validation (Priority: MEDIUM)

### 4.1 Input Validation

**Add validation at key entry points:**

1. **Config file validation**
   ```python
   # In ninjax_pipe.py

   def validate_config(self):
       """Validate configuration has all required fields with valid values"""
       required_fields = ["likelihood", "n_chains", "n_training_loops", ...]

       for field in required_fields:
           if field not in self.config:
               raise ValueError(f"Required config field missing: {field}")

       # Type validation
       if not isinstance(int(self.config["n_chains"]), int):
           raise ValueError("n_chains must be an integer")

       # Range validation
       if int(self.config["n_chains"]) <= 0:
           raise ValueError("n_chains must be positive")
   ```

2. **Injection parameter validation**
   ```python
   # In pipe_utils.py

   def validate_injection(injection: dict):
       """Validate injection parameters are physical"""
       for key, value in injection.items():
           if not jnp.isfinite(value):
               raise ValueError(f"Injection parameter {key}={value} is not finite!")

       # Physics constraints
       if "M_c" in injection and injection["M_c"] <= 0:
           raise ValueError(f"Chirp mass must be positive, got {injection['M_c']}")

       if "q" in injection and not (0 < injection["q"] <= 1):
           raise ValueError(f"Mass ratio must be in (0,1], got {injection['q']}")
   ```

3. **Transform pipeline validation**
   ```python
   # In ninjax_pipe.py

   def validate_transform_pipeline(self):
       """Ensure transforms don't introduce NaNs"""
       # Sample from prior
       test_sample = self.complete_prior.sample(jax.random.PRNGKey(0), 1)

       # Apply transforms
       transformed = test_sample.copy()
       for transform in self.likelihood_transforms:
           transformed = transform.forward(transformed)

           # Check for NaNs
           for key, value in transformed.items():
               if jnp.isnan(value).any():
                   raise ValueError(
                       f"Transform {transform} introduced NaNs in {key}"
                   )
   ```

### 4.2 Better Error Messages

**Current problem**: Many functions fail silently or with cryptic errors

**Improvements:**

1. **Config parsing errors** - show line numbers and context
2. **Missing PSD files** - show full path attempted and suggestions
3. **Waveform evaluation errors** - show which parameters caused failure
4. **Transform errors** - show which transform failed and on what parameters

---

## 5. FEATURES: New Capabilities (Priority: LOW-MEDIUM)

### 5.1 Checkpoint/Resume Functionality

**Use case**: Long runs (hours/days) need ability to resume if interrupted

**Implementation sketch:**
```python
# In analysis.py

def save_checkpoint(jim, outdir, iteration):
    """Save current sampler state to disk"""
    checkpoint_path = os.path.join(outdir, f"checkpoint_{iteration}.npz")

    np.savez(
        checkpoint_path,
        iteration=iteration,
        chains=jim.sampler.resources["positions_production"].data,
        log_prob=jim.sampler.resources["log_prob_production"].data,
        # ... other state
    )

def load_checkpoint(outdir):
    """Load most recent checkpoint"""
    checkpoints = glob.glob(os.path.join(outdir, "checkpoint_*.npz"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=os.path.getctime)
    return np.load(latest)
```

### 5.2 Real-Time Monitoring

**Use case**: Monitor sampling progress without waiting for completion

**Possible approaches:**
- TensorBoard integration (log metrics during training)
- Simple web dashboard (Flask + plotly)
- File-based progress tracking (write intermediate results every N iterations)

### 5.3 Sampling Diagnostics

**Add to output:**
- Effective sample size (ESS)
- Gelman-Rubin R-hat statistic
- Autocorrelation times
- Acceptance rate diagnostics

**Recommended library**: `arviz` (ArviZ - for Bayesian inference diagnostics)

```python
import arviz as az

# In analysis.py, after sampling
def compute_diagnostics(chains):
    """Compute convergence diagnostics"""
    # Convert to arviz InferenceData format
    inference_data = az.from_dict(
        posterior={
            key: chains[key] for key in chains.keys()
        }
    )

    # Compute diagnostics
    summary = az.summary(inference_data)
    ess = az.ess(inference_data)
    rhat = az.rhat(inference_data)

    return summary, ess, rhat
```

---

## 6. DOCUMENTATION: Improve User Experience (Priority: MEDIUM)

### 6.1 Example Gallery

**Create comprehensive examples:**

```
ninjax/examples/
├── example_1_basic_bns/              ✅ DONE
│   ├── README.md                     # What this example demonstrates
│   ├── config.ini
│   └── prior.prior
├── example_2_overlapping_signals/    ⚠️ TODO
│   └── ...
├── example_3_real_data/              ⚠️ TODO
│   └── ...
├── example_4_precessing_bbh/         ⚠️ TODO
│   └── ...
└── example_5_custom_prior/           ⚠️ TODO (using NFPrior)
    └── ...
```

Each example should include:
- `README.md` explaining what it demonstrates
- Expected runtime and resource requirements
- Expected results (plots, parameter estimates)
- Scientific context (what physics is being tested)

### 6.2 API Documentation

**Generate docs with Sphinx:**

```bash
# Setup
cd ninjax
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs

# Configure docs/conf.py to autodoc
# Build
sphinx-build -b html docs docs/_build
```

**Document:**
- All public classes and methods
- Config file options (auto-generate from defaults)
- Prior file syntax
- Transform pipeline options

### 6.3 User Guide

**Topics to cover:**
1. Installation and setup
2. Config file format and options
3. Prior specification syntax
4. Understanding output files
5. Troubleshooting common errors
6. Performance tuning guide

---

## 7. PERFORMANCE: Optimization Opportunities (Priority: LOW)

### 7.1 Profiling

**Tools:**
- `jax.profiler` for GPU profiling
- `cProfile` for CPU profiling
- `memory_profiler` for memory usage

**Target areas:**
- Likelihood evaluation (should be JIT-compiled)
- Waveform generation (check if properly vectorized)
- Transform application (can they be batched?)

### 7.2 Caching

**Opportunities:**
- Cache waveform evaluations (if parameters don't change much)
- Cache PSD interpolation
- Cache transform results (if applicable)

---

## 8. PRIORITY RANKING

### IMMEDIATE (Do in next session):
1. ✅ Add TODO comments throughout codebase (DONE)
2. **Remove deprecated code** (`LikelihoodWithTransforms`, unused transform functions)
3. **Fix NFPrior hardcoding** (make dimension-agnostic)
4. **Add unit tests for NFPrior**

### SHORT-TERM (Next 1-2 weeks):
1. **Test BaseTransientLikelihoodFD** (create example)
2. **Implement automatic waveform parameter extraction**
3. **Add config-driven transform setup**
4. **Split pipe_utils.py** into organized modules

### MEDIUM-TERM (Next month):
1. **Test overlapping injections** (create example, refactor to reduce duplication)
2. **Test real data loading** (create example with LIGO data)
3. **Add comprehensive input validation**
4. **Implement checkpoint/resume**

### LONG-TERM (When needed):
1. **Add sampling diagnostics** (ESS, R-hat)
2. **Create example gallery** (all use cases)
3. **Set up Sphinx documentation**
4. **Performance profiling and optimization**

---

## 9. ARCHITECTURAL CONSIDERATIONS

### 9.1 Separation of Concerns

**Current structure is reasonable, but could be improved:**

```
ninjax/
├── src/ninjax/
│   ├── core/                        # Core abstractions (NEW)
│   │   ├── pipeline.py              # Base pipeline class
│   │   ├── likelihood.py            # Custom likelihoods only
│   │   └── prior.py                 # Custom priors only
│   ├── pipes/                       # Concrete implementations
│   │   ├── gw_pipe.py               # GW-specific logic
│   │   └── ninjax_pipe.py           # Main orchestration
│   ├── utils/                       # Utilities (REFACTOR)
│   │   ├── plotting.py
│   │   ├── io_utils.py
│   │   ├── physics.py
│   │   └── injection.py
│   ├── transforms.py                # DEPRECATE or integrate with jim
│   ├── parser.py                    # Config parsing
│   └── analysis.py                  # Main script
└── test/                            # NEW - comprehensive tests
    ├── unit/
    └── integration/
```

### 9.2 Dependency Injection

**Consider making components more testable:**

```python
class GWPipe:
    def __init__(
        self,
        config: dict,
        prior: CombinePrior,
        waveform_factory: Callable,      # Injected
        detector_factory: Callable,      # Injected
        likelihood_class: Type[LikelihoodBase],  # Injected
    ):
        # More testable - can inject mocks
        pass
```

**Benefits**:
- Easier to test (mock dependencies)
- More flexible (swap implementations)
- Clearer dependencies

---

## 10. COLLABORATION WORKFLOW

### 10.1 Git Workflow

**Recommended branching strategy:**
```
main                    # Stable, tested code
├── develop             # Integration branch
│   ├── feature/xxx     # New features
│   ├── bugfix/yyy      # Bug fixes
│   └── refactor/zzz    # Code improvements
```

### 10.2 Code Review Checklist

Before merging:
- [ ] All TODOs addressed or documented
- [ ] Unit tests added for new code
- [ ] Integration test passes
- [ ] Documentation updated
- [ ] No hardcoded paths or assumptions
- [ ] Type hints added
- [ ] Linting passes (ruff, pyright)

---

## CONCLUSION

The ninjax package is now successfully migrated to the new jim API, but several areas need attention:

**Most Critical**:
1. Remove deprecated code to reduce confusion
2. Fix NFPrior hardcoding (very fragile)
3. Test untested features (BaseTransientLikelihoodFD, overlapping, etc.)

**Highest Impact**:
1. Automatic waveform parameter handling (eliminates manual errors)
2. Config-driven transforms (improves flexibility)
3. Comprehensive testing (ensures correctness)

**Best Practices**:
1. Add input validation everywhere
2. Improve error messages
3. Split large files
4. Document everything

Focus on the IMMEDIATE and SHORT-TERM items first. The MEDIUM-TERM and LONG-TERM items can wait until the core functionality is solid and well-tested.

Good luck with the next development phase! 🚀
