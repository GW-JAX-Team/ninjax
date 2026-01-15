# Example 3: Heterodyned Phase-Marginalized Likelihood

This example demonstrates the use of the **HeterodynedPhaseMarginalizedLikelihoodFD** likelihood class from jim. This likelihood combines two computational techniques:

1. **Phase Marginalization**: Analytically marginalizes over the coalescence phase parameter (`phase_c`), reducing the dimensionality of the sampling space by one. This is achieved by computing `log_i0(|<h|d>|)` instead of the standard matched filter SNR.

2. **Heterodyned Relative Binning**: Uses relative binning to speed up waveform evaluation by evaluating the waveform on a coarse frequency grid and interpolating. This dramatically reduces computational cost for long signals or high sampling rates.

## Configuration

The configuration is identical to `example_2` except for the likelihood choice:

- **Likelihood**: `HeterodynedPhaseMarginalizedLikelihoodFD`
- **Waveform**: TaylorF2 (post-Newtonian BNS)
- **Detectors**: H1, L1, V1
- **Injection**: Pre-existing injection with SNR 20-50
- **Relative binning**: 500 bins, using true parameters as reference

## When to Use This Likelihood

Use `HeterodynedPhaseMarginalizedLikelihoodFD` when:

1. You don't care about the coalescence phase (most GW inference applications)
2. You need fast waveform evaluation (long signals, high sampling rates, or many evaluations)
3. You have a good reference waveform (e.g., from a detection pipeline or a previous run)

## Expected Behavior

Compared to `example_2` (which does NOT marginalize over phase):

- **Sampling efficiency**: Should converge faster since we reduced dimensionality by 1
- **Posterior**: The `phase_c` parameter should NOT appear in output plots
- **Speed**: Similar speed to `example_2` since both use heterodyning
- **Results**: All other parameters (masses, spins, distance, sky location) should match

## Running the Example

```bash
# From this directory
ninjax_analysis .

# Or submit to SLURM cluster
sbatch submit.sh
```

## Comparison with Other Likelihoods

| Likelihood | Phase Marg? | Heterodyned? | Speed | Use Case |
|------------|-------------|--------------|-------|----------|
| `BaseTransientLikelihoodFD` | No | No | Slow | Small signals, low frequency |
| `PhaseMarginalizedLikelihoodFD` | Yes | No | Slow | When phase is unimportant |
| `HeterodynedTransientLikelihoodFD` | No | Yes | Fast | When phase matters |
| `HeterodynedPhaseMarginalizedLikelihoodFD` | Yes | Yes | Fast | **Recommended** for most applications |

## Implementation Details

From `jimgw/core/single_event/likelihood.py`:

- The phase-marginalized likelihood uses `log_i0(|complex_d_inner_h|)` to analytically marginalize over phase
- The heterodyned version evaluates waveforms at coarse frequency bins and uses linear interpolation
- Reference parameters can be provided or automatically found via optimization

## See Also

- `example_1`: Basic non-heterodyned likelihood (slower)
- `example_2`: Heterodyned likelihood WITHOUT phase marginalization
- Jim documentation: https://github.com/ThibeauWouters/jim
