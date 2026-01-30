
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import tmlsm.configs as configs
import numpy as np

def test_sweep_generation():
    print("Testing generate_sweep_configs...")
    
    # Test Omega sweep
    omega_configs = configs.generate_sweep_configs(
        param_name="omega",
        min_val=0.5,
        max_val=5.0,
        n_steps=10,
        fixed_val=1.0
    )
    
    assert len(omega_configs) == 10, f"Expected 10 configs, got {len(omega_configs)}"
    print(f"Generated {len(omega_configs)} configs for omega sweep.")
    
    first = omega_configs[0]
    last = omega_configs[-1]
    
    assert np.isclose(first.train_loadcases[0][1], 0.5), f"First omega should be 0.5, got {first.train_loadcases[0][1]}"
    assert np.isclose(last.train_loadcases[0][1], 5.0), f"Last omega should be 5.0, got {last.train_loadcases[0][1]}"
    assert first.train_loadcases[0][0] == 1.0, f"Amplitude should be fixed at 1.0, got {first.train_loadcases[0][0]}"
    
    print(f"First config: {first.name}, {first.train_loadcases}")
    print(f"Last config: {last.name}, {last.train_loadcases}")

    # Test Amplitude sweep
    A_configs = configs.generate_sweep_configs(
        param_name="A",
        min_val=1.0, 
        max_val=2.0,
        n_steps=3
    )
    assert len(A_configs) == 3
    assert np.isclose(A_configs[1].train_loadcases[0][0], 1.5)
    print(f"Generated {len(A_configs)} configs for A sweep.")

    print("\nSUCCESS: All tests passed!")

if __name__ == "__main__":
    test_sweep_generation()
