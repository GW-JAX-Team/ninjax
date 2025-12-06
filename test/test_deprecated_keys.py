"""
Test script to verify that deprecated configuration keys are properly rejected.

This script creates a test config with deprecated keys and verifies that
the NinjaxPipe raises a ValueError with helpful error messages.
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ninjax.pipes.ninjax_pipe import NinjaxPipe


def test_deprecated_keys():
    """Test that deprecated configuration keys trigger helpful errors."""

    # Create a temporary test directory
    test_dir = tempfile.mkdtemp(prefix='ninjax_test_')

    try:
        # Write a config file with deprecated keys
        config_content = """### Likelihood
likelihood = HeterodynedTransientLikelihoodFD

### GW inference
gw_injection = True
waveform_approximant = TaylorF2
ifos = H1,L1,V1
fmin = 20.0
fmax = 2048.0
trigger_time = 1187008882.4

### DEPRECATED KEYS (should cause errors)
eps_mass_matrix = 1e-4
n_loop_training = 50
n_loop_production = 20
max_samples = 5000
"""

        config_file = os.path.join(test_dir, 'config.ini')
        with open(config_file, 'w') as f:
            f.write(config_content)

        # Copy a valid prior file
        prior_src = os.path.join(os.path.dirname(__file__), '..', 'examples', 'example_1', 'prior.prior')
        prior_dst = os.path.join(test_dir, 'prior.prior')
        shutil.copy(prior_src, prior_dst)

        # Try to load the config - should fail with ValueError
        try:
            pipe = NinjaxPipe(test_dir)
            print("✗ FAILED: Deprecated keys were not caught!")
            return False
        except ValueError as e:
            error_msg = str(e)
            if 'deprecated' in error_msg.lower():
                print("✓ PASSED: Deprecated keys properly rejected!")
                print("\n✓ Error message received:")
                print("-" * 60)
                print(error_msg)
                print("-" * 60)

                # Check that all deprecated keys are mentioned
                expected_keys = ['eps_mass_matrix', 'n_loop_training', 'n_loop_production', 'max_samples']
                for key in expected_keys:
                    if key in error_msg:
                        print(f"✓ Key '{key}' mentioned in error")
                    else:
                        print(f"✗ Key '{key}' NOT mentioned in error")
                        return False

                return True
            else:
                print(f"✗ FAILED: Got ValueError but not about deprecated keys: {e}")
                return False

    finally:
        # Clean up
        shutil.rmtree(test_dir)


if __name__ == '__main__':
    success = test_deprecated_keys()
    sys.exit(0 if success else 1)
