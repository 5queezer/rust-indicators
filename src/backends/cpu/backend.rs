//! CPU Backend Implementation
//!
//! This module provides a pure CPU-based backend for technical indicator calculations.
//! The CPU backend is optimized for sequential processing and is ideal for smaller datasets
//! or when GPU acceleration is not available or beneficial.
//!
//! # Architecture Overview
//!
//! The CPU backend is designed with simplicity and reliability as core principles:
//!
//! ## Core Components
//!
//! - **CpuBackend**: Main backend implementation that executes all indicators on CPU
//! - **CPU Implementations**: Optimized sequential algorithms in [`crate::backends::cpu::implementations`]
//! - **Macro Integration**: Uses [`cpu_method!`] and [`extract_safe!`] macros for code reduction
//!
//! ## Design Philosophy
//!
//! The CPU backend follows these design principles:
//!
//! 1. **Sequential Optimization**: Algorithms are optimized for single-threaded execution
//! 2. **Memory Efficiency**: Minimal memory allocation and efficient data access patterns
//! 3. **Reliability**: Robust error handling and safe array operations
//! 4. **Simplicity**: Clean, maintainable code with minimal complexity
//!
//! # Performance Characteristics
//!
//! ## Strengths
//!
//! - **Low Overhead**: No GPU initialization or data transfer costs
//! - **Consistent Performance**: Predictable execution times regardless of data size
//! - **Memory Efficient**: Direct memory access without GPU memory management
//! - **Universal Compatibility**: Works on any system without GPU requirements
//!
//! ## Optimal Use Cases
//!
//! - **Small to Medium Datasets**: < 1000-2000 data points
//! - **Real-time Processing**: Low-latency requirements where GPU overhead is prohibitive
//! - **Development and Testing**: Consistent baseline for performance comparisons
//! - **Production Fallback**: Reliable fallback when GPU is unavailable
//!
//! # Supported Indicators
//!
//! The CPU backend provides optimized implementations for all supported indicators:
//!
//! ## Momentum Indicators
//! - **RSI**: Relative Strength Index - momentum oscillator (0-100 range)
//! - **Williams %R**: Momentum indicator similar to RSI (-100 to 0 range)
//!
//! ## Moving Averages
//! - **EMA**: Exponential Moving Average - weighted average favoring recent data
//! - **SMA**: Simple Moving Average - arithmetic mean over specified period
//!
//! ## Volatility Indicators
//! - **Bollinger Bands**: Statistical bands around moving average (upper, middle, lower)
//! - **ATR**: Average True Range - volatility measure
//!
//! ## Trend Analysis
//! - **CCI**: Commodity Channel Index - trend strength and direction
//!
//! ## Volume Analysis
//! - **VPIN**: Volume-synchronized Probability of Informed Trading
//!
//! ## Advanced Signal Processing
//! - **SuperSmoother**: Ehlers SuperSmoother filter for noise reduction
//! - **Hilbert Transform**: Complex signal analysis for cycle detection
//!
//! # Implementation Approach
//!
//! ## Macro-Based Code Generation
//!
//! The CPU backend leverages the [`cpu_method!`] macro to eliminate boilerplate:
//!
//! ```rust
//! use rust_indicators::cpu_method;
//!
//! cpu_method!(rsi, rsi_cpu, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
//! ```
//!
//! This approach provides:
//! - **Consistency**: Uniform error handling and parameter validation
//! - **Maintainability**: Single point of change for common patterns
//! - **Code Reduction**: ~2 lines per indicator reduced to 1 macro call
//!
//! ## Safe Array Operations
//!
//! All array operations use the [`extract_safe!`] macro for robust error handling:
//!
//! ```rust
//! use rust_indicators::extract_safe;
//! let buy_slice = extract_safe!(buy_array, "buy_volumes");
//! ```
//!
//! This ensures:
//! - **Memory Safety**: No unsafe array access or unwrap() calls
//! - **Clear Error Messages**: Descriptive error messages for debugging
//! - **Graceful Degradation**: Proper PyResult error propagation
//!
//! # Usage Example
//!
//! ```rust
//! use rust_indicators::backends::cpu::CpuBackend;
//! use rust_indicators::core::traits::IndicatorsBackend;
//!
//! // Create CPU backend (always available)
//! let backend = CpuBackend::new();
//!
//! // Calculate RSI for price data
//! // let rsi_result = backend.rsi(py, prices, 14)?;
//!
//! // All indicators use optimized CPU implementations
//! // Performance is consistent and predictable
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Error Handling
//!
//! The CPU backend provides comprehensive error handling:
//!
//! - **Array Validation**: Ensures input arrays are contiguous and valid
//! - **Parameter Validation**: Validates periods and other parameters in implementation functions
//! - **Memory Safety**: No unsafe operations or potential panics
//! - **Clear Diagnostics**: Descriptive error messages for troubleshooting

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
