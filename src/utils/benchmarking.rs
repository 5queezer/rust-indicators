//! Benchmarking utilities for VPIN and other indicators
//!
//! This module provides shared utility functions for benchmarking
//! indicator computations, eliminating code duplication across backends.

/// Benchmark VPIN computation using CPU kernel
/// 
/// This function encapsulates the CPU VPIN kernel call for benchmarking purposes.
/// It's used by the adaptive backend for performance calibration and by the GPU
/// backend as a fallback when GPU features are not available.
///
/// # Arguments
/// * `buy_volumes` - Slice of buy volume data
/// * `sell_volumes` - Slice of sell volume data  
/// * `window` - Window size for VPIN calculation
///
/// # Returns
/// Vector of VPIN values computed using the CPU kernel
pub fn benchmark_vpin_cpu(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    crate::backends::cpu::implementations::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}