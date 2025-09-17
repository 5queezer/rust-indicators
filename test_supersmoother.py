#!/usr/bin/env python3
"""
Quick test to verify SuperSmoother integration works correctly
"""

import numpy as np
import sys
import os

# Add the project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target/wheels'))

try:
    import rust_indicators
    
    # Create test data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    period = 10
    
    # Initialize RustTA
    ta = rust_indicators.RustTA()
    print(f"Using device: {ta.device()}")
    
    # Test SuperSmoother
    result = ta.supersmoother(data, period)
    print(f"Input data: {data}")
    print(f"SuperSmoother result: {result}")
    print(f"Result shape: {result.shape}")
    
    # Verify the result has the same length as input
    assert len(result) == len(data), f"Expected length {len(data)}, got {len(result)}"
    
    # Verify first two values are 0 (initialization)
    assert result[0] == 0.0, f"Expected first value to be 0.0, got {result[0]}"
    assert result[1] == 0.0, f"Expected second value to be 0.0, got {result[1]}"
    
    # Verify we have non-zero values after index 2
    assert any(result[2:] != 0.0), "Expected non-zero values after index 2"
    
    print("✅ SuperSmoother integration test passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: You may need to build the Python extension first with 'maturin develop'")
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)