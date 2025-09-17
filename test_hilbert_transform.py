#!/usr/bin/env python3
"""
Test script for Hilbert Transform implementation
"""

import numpy as np
import sys
import os

# Add the current directory to Python path to import rust_indicators
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import rust_indicators

    # Create test data - a simple sine wave with trend
    n_points = 100
    t = np.linspace(0, 4*np.pi, n_points)

    # Create a composite signal: trend + sine wave + noise
    trend = 0.01 * t
    sine_wave = np.sin(t)
    noise = 0.1 * np.random.randn(n_points)
    test_data = trend + sine_wave + noise

    print(f"Testing Hilbert Transform with {n_points} data points")
    print(f"Test data range: [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Initialize RustTA
    ta = rust_indicators.RustTA()
    print(f"Using device: {ta.device()}")

    # Test different lp_period values
    test_periods = [10, 14, 20]

    for lp_period in test_periods:
        print(f"\n--- Testing with lp_period = {lp_period} ---")

        try:
            # Test Hilbert Transform
            real_component, imaginary_component = ta.hilbert_transform(test_data, lp_period)

            print(f"Real component shape: {real_component.shape}")
            print(f"Imaginary component shape: {imaginary_component.shape}")

            # Verify the results have the same length as input
            assert len(real_component) == len(test_data), f"Real component length mismatch: {len(real_component)} vs {len(test_data)}"
            assert len(imaginary_component) == len(test_data), f"Imaginary component length mismatch: {len(imaginary_component)} vs {len(test_data)}"

            # Check for reasonable values (not all zeros after initial period)
            non_zero_real = np.count_nonzero(real_component[50:])  # Skip initial period
            non_zero_imag = np.count_nonzero(imaginary_component[50:])

            print(f"Non-zero real values (after index 50): {non_zero_real}")
            print(f"Non-zero imaginary values (after index 50): {non_zero_imag}")

            # Calculate instantaneous amplitude and phase
            amplitude = np.sqrt(real_component**2 + imaginary_component**2)
            phase = np.arctan2(imaginary_component, real_component)

            print(f"Amplitude range: [{amplitude.min():.6f}, {amplitude.max():.6f}]")
            print(f"Phase range: [{phase.min():.6f}, {phase.max():.6f}]")

            # Basic sanity checks
            assert not np.all(real_component == 0), "Real component is all zeros"
            assert not np.all(imaginary_component == 0), "Imaginary component is all zeros"
            assert np.all(np.isfinite(real_component)), "Real component contains non-finite values"
            assert np.all(np.isfinite(imaginary_component)), "Imaginary component contains non-finite values"

            print(f"✅ Hilbert Transform test passed for lp_period = {lp_period}")

        except Exception as e:
            print(f"❌ Hilbert Transform test failed for lp_period = {lp_period}: {e}")
            raise

    print("\n🎉 All Hilbert Transform tests passed!")

    # Performance test with larger dataset
    print("\n--- Performance Test ---")
    large_data = np.random.randn(10000)

    import time
    start_time = time.time()
    real_large, imag_large = ta.hilbert_transform(large_data, 14)
    end_time = time.time()

    print(f"Large dataset ({len(large_data)} points) processed in {(end_time - start_time)*1000:.2f} ms")
    print(f"Performance target (<20ms for 10K points): {'✅ PASSED' if (end_time - start_time)*1000 < 20 else '⚠️ SLOWER THAN TARGET'}")

except ImportError as e:
    print(f"❌ Failed to import rust_indicators: {e}")
    print("Make sure to build the library first with: maturin develop")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
