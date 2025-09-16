use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::core::traits::IndicatorsBackend;
use crate::utils::backend_selection;

#[pyclass]
pub struct RustTA {
    backend: Box<dyn IndicatorsBackend>,
    device: &'static str,
}

impl RustTA {
    fn select_backend() -> (Box<dyn IndicatorsBackend>, &'static str) {
        backend_selection::select_backend()
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

    /// Volume-synchronized Probability of Informed Trading (VPIN)
    fn vpin(
        &self,
        py: Python,
        buy_volumes: PyReadonlyArray1<f64>,
        sell_volumes: PyReadonlyArray1<f64>,
        window: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        self.backend.vpin(py, buy_volumes, sell_volumes, window)
    }
}