//! CPU Backend Implementation
//!
//! This module provides a pure CPU-based backend for technical indicator calculations.
//! The CPU backend is optimized for sequential processing and is ideal for smaller datasets
//! or when GPU acceleration is not available or beneficial.

use crate::core::traits::IndicatorsBackend;
use crate::{cpu_method, extract_safe};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// CPU Backend for Technical Indicator Calculations
///
/// A pure CPU-based backend that provides reliable, sequential processing
/// for all supported technical indicators. Optimized for smaller datasets
/// and scenarios where GPU acceleration is not available or beneficial.
///
/// # Performance Profile
///
/// - **Optimal Range**: 100-2000 data points
/// - **Overhead**: Minimal (no GPU initialization)
/// - **Memory Usage**: Efficient direct memory access
/// - **Latency**: Low and consistent
///
/// # Thread Safety
///
/// The CPU backend is thread-safe and can be used concurrently across
/// multiple threads without synchronization concerns.
pub struct CpuBackend;

impl CpuBackend {
    /// Creates a new CPU backend instance
    ///
    /// The CPU backend is always available and requires no initialization.
    /// This constructor never fails and provides a reliable fallback option.
    ///
    /// # Returns
    ///
    /// A new `CpuBackend` instance ready for indicator calculations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rust_indicators::backends::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// // Backend is immediately ready for use
    /// ```
    pub fn new() -> Self {
        CpuBackend
    }
}

impl IndicatorsBackend for CpuBackend {
    cpu_method!(rsi, rsi_cpu, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(ema, ema_cpu, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(sma, sma_cpu, (values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(bollinger_bands, bollinger_bands_cpu, (prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)>);
    cpu_method!(atr, atr_cpu, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(williams_r, williams_r_cpu, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(cci, cci_cpu, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);

    /// Calculate Volume-synchronized Probability of Informed Trading (VPIN)
    ///
    /// VPIN is a volume-based indicator that measures the probability of informed trading
    /// by analyzing the imbalance between buy and sell volumes over a rolling window.
    ///
    /// # Parameters
    ///
    /// - `py`: Python context for PyO3 operations
    /// - `buy_volumes`: Array of buy volume data
    /// - `sell_volumes`: Array of sell volume data  
    /// - `window`: Rolling window size for VPIN calculation
    ///
    /// # Returns
    ///
    /// `PyResult<Py<PyArray1<f64>>>` containing the VPIN values
    ///
    /// # Performance Notes
    ///
    /// - **CPU Implementation**: Uses optimized sequential algorithm
    /// - **Complexity**: O(n * window) where n is data length
    /// - **Memory**: Efficient in-place calculations with minimal allocation
    ///
    /// # Example
    ///
    /// ```python
    /// # Python usage
    /// vpin_values = backend.vpin(buy_volumes, sell_volumes, window=50)
    /// ```
    fn vpin<'py>(
        &self,
        py: Python<'py>,
        buy_volumes: PyReadonlyArray1<'py, f64>,
        sell_volumes: PyReadonlyArray1<'py, f64>,
        window: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let buy_array = buy_volumes.as_array();
        let sell_array = sell_volumes.as_array();
        let buy_slice = extract_safe!(buy_array, "buy_volumes");
        let sell_slice = extract_safe!(sell_array, "sell_volumes");
        let results =
            crate::backends::cpu::implementations::vpin_cpu_kernel(buy_slice, sell_slice, window);
        Ok(PyArray1::from_vec(py, results).to_owned().into())
    }

    cpu_method!(supersmoother, supersmoother_cpu, (data: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    cpu_method!(hilbert_transform, hilbert_transform_cpu, (data: PyReadonlyArray1<'py, f64>, lp_period: usize) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>);
}
