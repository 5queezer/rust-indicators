#[cfg(feature = "gpu")]
use cubecl::wgpu::WgpuDevice;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::backend::IndicatorsBackend;

pub struct GpuBackend {
    #[cfg(feature = "gpu")]
    device: WgpuDevice,
}

impl GpuBackend {
    pub fn try_new() -> anyhow::Result<Self> {
        #[cfg(feature = "gpu")]
        {
            let device = WgpuDevice::default();
            Ok(GpuBackend { device })
        }
        #[cfg(not(feature = "gpu"))]
        {
            anyhow::bail!("gpu feature not compiled");
        }
    }
}

impl IndicatorsBackend for GpuBackend {
    fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::rsi_cpu(py, prices, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!("gpu backend without feature") }
    }
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::ema_cpu(py, prices, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::sma_cpu(py, values, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::bollinger_bands_cpu(py, prices, period, std_dev) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::atr_cpu(py, high, low, close, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::williams_r_cpu(py, high, low, close, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        { crate::cpu_impls::cci_cpu(py, high, low, close, period) }
        #[cfg(not(feature = "gpu"))]
        { unreachable!() }
    }
    fn vpin<'py>(
        &self,
        py: Python<'py>,
        buy_volumes: PyReadonlyArray1<'py, f64>,
        sell_volumes: PyReadonlyArray1<'py, f64>,
        window: usize
    ) -> PyResult<Py<PyArray1<f64>>> {
        #[cfg(feature = "gpu")]
        {
            crate::vpin_kernel::vpin_kernel_wrapper(py, &self.device, buy_volumes, sell_volumes, window)
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("gpu backend without feature")
        }
    }
}
