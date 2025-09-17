use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
    fn log_returns(
        &self,
        py: Python,
        prices: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
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
                    let variance: f64 =
                        window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / denom;

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
