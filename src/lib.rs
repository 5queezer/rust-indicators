// File: rust_indicators/src/lib.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

// Conditionally use mimalloc when the feature is enabled
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Rust-powered technical analysis indicators for FreqTrade
/// Provides 10-100x performance improvement over Python equivalents
pub mod backend;
pub mod backend_cpu;
#[cfg(feature = "gpu")]
pub mod backend_gpu;
pub mod cpu_impls;

use backend::IndicatorsBackend;
use backend_cpu::CpuBackend;
#[cfg(feature = "gpu")]
use backend_gpu::GpuBackend;

#[pyclass]
pub struct RustTA {
    backend: Box<dyn IndicatorsBackend>,
    device: &'static str,
}

impl RustTA {
    fn select_backend() -> (Box<dyn IndicatorsBackend>, &'static str) {
        let forced = std::env::var("RUST_INDICATORS_DEVICE").ok();
        if forced.as_deref() == Some("cpu") {
            return (Box::new(CpuBackend::new()), "cpu");
        }
        #[cfg(feature = "gpu")]
        {
            if forced.as_deref() == Some("gpu") {
                match GpuBackend::try_new() {
                    Ok(g) => return (Box::new(g), "gpu"),
                    Err(e) => {
                        eprintln!("[rust_indicators] GPU requested but unavailable: {e}. Falling back to CPU.");
                        return (Box::new(CpuBackend::new()), "cpu");
                    }
                }
            }
            if let Ok(g) = GpuBackend::try_new() { return (Box::new(g), "gpu"); }
            else { eprintln!("[rust_indicators] No usable GPU; using CPU."); }
        }
        (Box::new(CpuBackend::new()), "cpu")
    }
}

#[pymethods]
impl RustTA {
    #[new]
    fn new() -> Self {
        let (backend, device) = Self::select_backend();
        RustTA { backend, device }
    }

    /// Which device is in use ("cpu" or "gpu")
    fn device(&self) -> &str { self.device }

    /// RSI calculation - optimized implementation
    fn rsi(&self, py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.rsi(py, prices, period)
    }

    /// Exponential Moving Average - ultra-fast implementation
    fn ema(&self, py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.ema(py, prices, period)
    }

    /// Simple Moving Average
    fn sma(&self, py: Python, values: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.sma(py, values, period)
    }

    /// Bollinger Bands calculation
    fn bollinger_bands(
        &self,
        py: Python,
        prices: PyReadonlyArray1<f64>,
        period: usize,
        std_dev: f64,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        self.backend.bollinger_bands(py, prices, period, std_dev)
    }

    /// Average True Range
    fn atr(
        &self,
        py: Python,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.atr(py, high, low, close, period)
    }

    /// Williams %R
    fn williams_r(
        &self,
        py: Python,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.williams_r(py, high, low, close, period)
    }

    /// Commodity Channel Index
    fn cci(
        &self,
        py: Python,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        period: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.cci(py, high, low, close, period)
    }
}

/// Feature engineering module
#[pyclass]
pub struct RustFeatures;

#[pymethods]
impl RustFeatures {
    #[new]
    fn new() -> Self {
        RustFeatures
    }

    /// Fractional differentiation for stationarity with memory preservation
    fn fractional_diff(
        &self,
        py: Python,
        series: PyReadonlyArray1<f64>,
        d: f64,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let series = series.as_array();
        let len = series.len();
        let mut result = vec![f64::NAN; len];

        // Simple fractional differentiation approximation
        // For production, use full FFD implementation
        for i in 1..len {
            let diff = series[i] - series[i - 1];
            let frac_diff = diff * d + series[i - 1] * (1.0 - d);
            result[i] = frac_diff;
        }

        Ok(PyArray1::from_vec(py, result).to_owned().into())
    }

    /// Log returns calculation
    fn log_returns(&self, py: Python, prices: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let prices = prices.as_array();
        let len = prices.len();
        let mut result = vec![f64::NAN; len];

        for i in 1..len {
            if prices[i - 1] > 0.0 && prices[i] > 0.0 {
                result[i] = (prices[i] / prices[i - 1]).ln();
            }
        }

        Ok(PyArray1::from_vec(py, result).to_owned().into())
    }

    /// Rolling volatility (standard deviation of returns)
    fn rolling_volatility(
        &self,
        py: Python,
        returns: PyReadonlyArray1<f64>,
        window: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let returns = returns.as_array();
        let len = returns.len();
        let mut result = vec![f64::NAN; len];

        if window == 0 {
            return Ok(PyArray1::from_vec(py, result).to_owned().into());
        }

        for i in 0..len {
            if i + 1 >= window {
                let start_idx = i + 1 - window;
                let window_data: Vec<f64> = (start_idx..=i)
                    .map(|j| returns[j])
                    .filter(|x| x.is_finite())
                    .collect();

                if !window_data.is_empty() {
                    let mean: f64 = window_data.iter().sum::<f64>() / window_data.len() as f64;
                    let denom = (window_data.len() - 1).max(1) as f64;
                    let variance: f64 = window_data
                        .iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / denom;

                    result[i] = variance.sqrt();
                }
            }
        }

        Ok(PyArray1::from_vec(py, result).to_owned().into())
    }

    /// Volatility regime detection (0 = low vol, 1 = high vol)
    fn volatility_regime(
        &self,
        py: Python,
        volatility: PyReadonlyArray1<f64>,
        threshold: f64,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let volatility = volatility.as_array();
        let len = volatility.len();
        let mut result = vec![f64::NAN; len];

        for i in 0..len {
            if volatility[i].is_finite() {
                result[i] = if volatility[i] > threshold { 1.0 } else { 0.0 };
            }
        }

        Ok(PyArray1::from_vec(py, result).to_owned().into())
    }

    /// Volume bars - sample when cumulative volume reaches threshold
    fn volume_bars(
        &self,
        py: Python,
        prices: PyReadonlyArray1<f64>,
        volumes: PyReadonlyArray1<f64>,
        threshold: f64,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let prices = prices.as_array();
        let volumes = volumes.as_array();
        let len = prices.len().min(volumes.len());
        let mut result = vec![0.0; len];

        let mut cumulative_volume = 0.0;
        let mut bar_count = 0.0;

        for i in 0..len {
            cumulative_volume += volumes[i];

            if cumulative_volume >= threshold {
                bar_count += 1.0;
                cumulative_volume = 0.0; // Reset
            }

            result[i] = bar_count;
        }

        Ok(PyArray1::from_vec(py, result).to_owned().into())
    }
}

/// Lightweight ML model for meta-labeling
#[pyclass]
pub struct RustMLModel {
    // Simple linear model weights for demonstration
    weights: Vec<f64>,
    bias: f64,
    trained: bool,
}

#[pymethods]
impl RustMLModel {
    #[new]
    fn new() -> Self {
        RustMLModel {
            weights: vec![0.1, -0.05, 0.15, 0.2, -0.1, 0.25, 0.3], // 7 features
            bias: 0.5,
            trained: false,
        }
    }

    /// Simple prediction using linear model + sigmoid activation
    fn predict(&self, py: Python, features: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let features = features.as_array();
        let n_features = self.weights.len();
        let n_samples = features.len() / n_features;

        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start_idx = i * n_features;
            let end_idx = start_idx + n_features;

            if end_idx <= features.len() {
                // Linear combination
                let mut linear_output = self.bias;
                for j in 0..n_features.min(end_idx - start_idx) {
                    let feature_val = features[start_idx + j];
                    if feature_val.is_finite() {
                        linear_output += self.weights[j] * feature_val;
                    }
                }

                // Sigmoid activation for probability output
                let probability = 1.0 / (1.0 + (-linear_output).exp());
                predictions.push(probability.clamp(0.01, 0.99)); // Avoid extreme values
            } else {
                predictions.push(0.5); // Neutral when insufficient features
            }
        }

        Ok(PyArray1::from_vec(py, predictions).to_owned().into())
    }

    /// Check if model is trained
    fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get model weights for inspection
    fn get_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_vec(py, self.weights.clone()).to_owned().into()
    }
}

/// Python module definition
#[pymodule]
mod rust_indicators {
    #[pymodule_export]
    use super::RustTA;
    
    #[pymodule_export]
    use super::RustFeatures;
    
    #[pymodule_export]
    use super::RustMLModel;
}
