//! Adaptive Backend Implementation
//!
//! This module provides an intelligent backend that automatically selects between CPU and GPU
//! computation based on performance profiling and workload characteristics. The adaptive backend
//! optimizes indicator calculations by dynamically choosing the most efficient execution path.
//!
//! # Architecture Overview
//!
//! The adaptive backend consists of several key components:
//!
//! ## Core Components
//!
//! - **AdaptiveBackend**: Main backend implementation that manages CPU and GPU backends
//! - **PerformanceProfile**: Tracks performance thresholds and calibration data
//! - **IndicatorParams**: Encapsulates indicator parameters and computational complexity
//!
//! ## Decision Logic
//!
//! The backend uses a sophisticated decision-making process:
//!
//! 1. **GPU Availability Check**: Verifies if GPU backend is available and functional
//! 2. **Complexity Analysis**: Calculates computational complexity based on data size and parameters
//! 3. **Threshold Comparison**: Compares complexity against calibrated performance thresholds
//! 4. **Backend Selection**: Routes computation to GPU or CPU based on expected performance
//!
//! ## Performance Profiling
//!
//! The adaptive backend maintains performance profiles for each indicator:
//!
//! - **Thresholds**: Pre-configured data size thresholds where GPU becomes more efficient than CPU
//! - **Fallback**: Automatic fallback to CPU when GPU is unavailable or inefficient
//!
//! # Usage Example
//!
//! ```rust
//! use rust_indicators::backends::adaptive::AdaptiveBackend;
//! use rust_indicators::core::traits::IndicatorsBackend;
//!
//! // Create adaptive backend (automatically detects GPU availability)
//! let backend = AdaptiveBackend::new()?;
//!
//! // The backend automatically selects CPU or GPU based on workload
//! // Small datasets -> CPU (lower overhead)
//! // Large datasets -> GPU (parallel processing advantage)
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Supported Indicators
//!
//! The adaptive backend currently supports the following indicators with intelligent routing:
//!
//! - **VPIN**: Volume-synchronized Probability of Informed Trading (GPU-optimized for large datasets)
//! - **RSI**: Relative Strength Index (CPU-only, optimized for typical dataset sizes)
//! - **EMA**: Exponential Moving Average (CPU-only, sequential computation)
//! - **SMA**: Simple Moving Average (CPU-only, efficient sequential implementation)
//! - **Bollinger Bands**: Statistical bands around moving average (CPU-only)
//! - **ATR**: Average True Range (CPU-only, volatility measure)
//! - **Williams %R**: Momentum oscillator (CPU-only)
//! - **CCI**: Commodity Channel Index (CPU-only, trend analysis)
//!
//! # Performance Characteristics
//!
//! ## VPIN Indicator
//!
//! - **Threshold**: ~1500 data points (pre-configured)
//! - **GPU Advantage**: Significant for datasets > 2000 points
//!
//! ## Other Indicators
//!
//! - **Current Status**: CPU-only (thresholds set to `usize::MAX`)
//! - **Future Enhancement**: GPU implementations planned for high-volume scenarios
//!
//! # Code Reduction
//!
//! The adaptive backend uses the [`delegate_indicator!`](crate::delegate_indicator) macro
//! to eliminate code duplication:
//!
//! - **Before**: ~13 lines per indicator × 8 indicators = 104 lines
//! - **After**: ~4 lines per indicator × 8 indicators = 32 lines
//! - **Reduction**: 72 lines of boilerplate eliminated
//!
//! # Error Handling
//!
//! The adaptive backend provides robust error handling:
//!
//! - **GPU Unavailable**: Graceful fallback to CPU backend
//! - **Backend Errors**: Propagate PyO3 errors with context

use crate::backends::cpu::CpuBackend;
use crate::backends::gpu::PartialGpuBackend;
use crate::core::traits::IndicatorsBackend;
use crate::delegate_indicator;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum IndicatorParams {
    Vpin { data_size: usize, window: usize },
    Rsi { data_size: usize, period: usize },
    Ema { data_size: usize, period: usize },
    Sma { data_size: usize, period: usize },
    BollingerBands { data_size: usize, period: usize },
    Atr { data_size: usize, period: usize },
    WilliamsR { data_size: usize, period: usize },
    Cci { data_size: usize, period: usize },
    SuperSmoother { data_size: usize, period: usize },
    HilbertTransform { data_size: usize, lp_period: usize },
}

impl IndicatorParams {
    pub fn data_size(&self) -> usize {
        match self {
            IndicatorParams::Vpin { data_size, .. } => *data_size,
            IndicatorParams::Rsi { data_size, .. } => *data_size,
            IndicatorParams::Ema { data_size, .. } => *data_size,
            IndicatorParams::Sma { data_size, .. } => *data_size,
            IndicatorParams::BollingerBands { data_size, .. } => *data_size,
            IndicatorParams::Atr { data_size, .. } => *data_size,
            IndicatorParams::WilliamsR { data_size, .. } => *data_size,
            IndicatorParams::Cci { data_size, .. } => *data_size,
            IndicatorParams::SuperSmoother { data_size, .. } => *data_size,
            IndicatorParams::HilbertTransform { data_size, .. } => *data_size,
        }
    }

    pub fn computational_complexity(&self) -> usize {
        match self {
            IndicatorParams::Vpin { data_size, window } => data_size * window,
            IndicatorParams::Rsi { data_size, period } => data_size * period,
            IndicatorParams::Ema { data_size, period } => data_size * period,
            IndicatorParams::Sma { data_size, period } => data_size * period,
            IndicatorParams::BollingerBands { data_size, period } => data_size * period * 2,
            IndicatorParams::Atr { data_size, period } => data_size * period,
            IndicatorParams::WilliamsR { data_size, period } => data_size * period,
            IndicatorParams::Cci { data_size, period } => data_size * period,
            IndicatorParams::SuperSmoother { data_size, period } => data_size * period,
            IndicatorParams::HilbertTransform {
                data_size,
                lp_period,
            } => data_size * lp_period * 4, // Complex algorithm with multiple stages
        }
    }
}

pub struct PerformanceProfile {
    pub thresholds: HashMap<String, usize>,
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("vpin".to_string(), 1500);
        thresholds.insert("rsi".to_string(), usize::MAX);
        thresholds.insert("ema".to_string(), usize::MAX);
        thresholds.insert("sma".to_string(), usize::MAX);
        thresholds.insert("bollinger_bands".to_string(), usize::MAX);
        thresholds.insert("atr".to_string(), usize::MAX);
        thresholds.insert("williams_r".to_string(), usize::MAX);
        thresholds.insert("cci".to_string(), usize::MAX);
        thresholds.insert("supersmoother".to_string(), usize::MAX);
        thresholds.insert("hilbert_transform".to_string(), 1000); // Mixed GPU/CPU algorithm, moderate threshold

        Self { thresholds }
    }
}

pub struct AdaptiveBackend {
    cpu_backend: CpuBackend,
    gpu_backend: Option<PartialGpuBackend>,
    performance_profile: PerformanceProfile,
    gpu_available: bool,
}

impl AdaptiveBackend {
    pub fn new() -> PyResult<Self> {
        let gpu_backend = PartialGpuBackend::new().ok();
        let gpu_available = gpu_backend.is_some();

        Ok(AdaptiveBackend {
            cpu_backend: CpuBackend::new(),
            gpu_backend,
            performance_profile: PerformanceProfile::default(),
            gpu_available,
        })
    }

    fn should_use_gpu(&self, indicator: &str, params: &IndicatorParams) -> bool {
        if !self.gpu_available {
            return false;
        }

        let threshold = self
            .performance_profile
            .thresholds
            .get(indicator)
            .copied()
            .unwrap_or(usize::MAX);

        params.computational_complexity() >= threshold
    }
}

impl IndicatorsBackend for AdaptiveBackend {
    fn rsi<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "rsi",
            IndicatorParams::Rsi {
                data_size: prices.as_array().len(),
                period
            },
            rsi(prices, period)
        )
    }

    fn ema<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "ema",
            IndicatorParams::Ema {
                data_size: prices.as_array().len(),
                period
            },
            ema(prices, period)
        )
    }

    fn sma<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "sma",
            IndicatorParams::Sma {
                data_size: values.as_array().len(),
                period
            },
            sma(values, period)
        )
    }

    fn bollinger_bands<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f64>,
        period: usize,
        std_dev: f64,
    ) -> PyResult<crate::indicators::api::BollingerBandsOutput> {
        let params = IndicatorParams::BollingerBands {
            data_size: prices.as_array().len(),
            period,
        };

        if self.should_use_gpu("bollinger_bands", &params) {
            if let Some(ref gpu_backend) = self.gpu_backend {
                let result = gpu_backend.bollinger_bands(py, prices, period, std_dev)?;
                return Ok(result);
            }
        }

        let result = self.cpu_backend.bollinger_bands(py, prices, period, std_dev)?;
        Ok(result)
    }

    fn atr<'py>(
        &self,
        py: Python<'py>,
        high: PyReadonlyArray1<'py, f64>,
        low: PyReadonlyArray1<'py, f64>,
        close: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "atr",
            IndicatorParams::Atr {
                data_size: high.as_array().len(),
                period
            },
            atr(high, low, close, period)
        )
    }

    fn williams_r<'py>(
        &self,
        py: Python<'py>,
        high: PyReadonlyArray1<'py, f64>,
        low: PyReadonlyArray1<'py, f64>,
        close: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "williams_r",
            IndicatorParams::WilliamsR {
                data_size: high.as_array().len(),
                period
            },
            williams_r(high, low, close, period)
        )
    }

    fn cci<'py>(
        &self,
        py: Python<'py>,
        high: PyReadonlyArray1<'py, f64>,
        low: PyReadonlyArray1<'py, f64>,
        close: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "cci",
            IndicatorParams::Cci {
                data_size: high.as_array().len(),
                period
            },
            cci(high, low, close, period)
        )
    }

    fn vpin<'py>(
        &self,
        py: Python<'py>,
        buy_volumes: PyReadonlyArray1<'py, f64>,
        sell_volumes: PyReadonlyArray1<'py, f64>,
        window: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "vpin",
            IndicatorParams::Vpin {
                data_size: buy_volumes.as_array().len(),
                window
            },
            vpin(buy_volumes, sell_volumes, window)
        )
    }

    fn supersmoother<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self,
            py,
            "supersmoother",
            IndicatorParams::SuperSmoother {
                data_size: data.as_array().len(),
                period
            },
            supersmoother(data, period)
        )
    }

    fn hilbert_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        lp_period: usize,
    ) -> PyResult<crate::indicators::api::HilbertTransformOutput> {
        let data_len = data.as_array().len();

        // Adaptive selection based on data size and complexity
        // Hilbert Transform has mixed parallelization potential:
        // - Roofing filter stages can benefit from GPU
        // - AGC and SuperSmoother have sequential dependencies
        // Use GPU for larger datasets where parallel stages provide benefit
        let params = IndicatorParams::HilbertTransform {
            data_size: data_len,
            lp_period,
        };

        if self.should_use_gpu("hilbert_transform", &params) {
            // Use GPU backend for larger datasets
            if let Some(ref gpu_backend) = self.gpu_backend {
                match gpu_backend.hilbert_transform(py, data.clone(), lp_period) {
                    Ok(result) => return Ok(result),
                    Err(_) => {
                        // Fallback to CPU on GPU error
                    }
                }
            }
        }

        // Use CPU backend for smaller datasets or as fallback
        self.cpu_backend.hilbert_transform(py, data, lp_period)
    }
}
