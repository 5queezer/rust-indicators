use crate::backend::IndicatorsBackend;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

pub struct CpuBackend;
impl CpuBackend { pub fn new() -> Self { CpuBackend } }

impl IndicatorsBackend for CpuBackend {
    fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::rsi_cpu(py, prices, period)
    }
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::ema_cpu(py, prices, period)
    }
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::sma_cpu(py, values, period)
    }
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        crate::cpu_impls::bollinger_bands_cpu(py, prices, period, std_dev)
    }
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::atr_cpu(py, high, low, close, period)
    }
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::williams_r_cpu(py, high, low, close, period)
    }
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::cci_cpu(py, high, low, close, period)
    }
}
