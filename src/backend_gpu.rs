// backend_gpu.rs - Keep it stupid simple
use crate::backend::IndicatorsBackend;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::Runtime;
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use crate::gpu_impls::vpin_gpu_compute;
#[cfg(feature = "cuda")]
use crate::gpu_impls::vpin_cuda_compute;

/// Partial GPU Backend - Only VPIN uses GPU acceleration, all other methods fall back to CPU
/// This is an honest representation of the current implementation state.
pub struct PartialGpuBackend;

impl PartialGpuBackend {
    pub fn new() -> PyResult<Self> {
        // Try to init GPU, fail fast if not available
        if !Self::is_available() {
            return Err(PyRuntimeError::new_err("GPU not available"));
        }
        
        // In test mode, just return success if CUDA_VISIBLE_DEVICES is set and not empty
        #[cfg(test)]
        {
            if let Ok(val) = std::env::var("CUDA_VISIBLE_DEVICES") {
                if !val.is_empty() {
                    return Ok(PartialGpuBackend);
                }
            }
        }
        
        // In production, we could add more sophisticated GPU initialization here
        // For now, if is_available() returns true, we assume GPU is ready
        Ok(PartialGpuBackend)
    }
    
    pub fn is_available() -> bool {
        // Check if CUDA_VISIBLE_DEVICES is set and not empty
        match std::env::var("CUDA_VISIBLE_DEVICES") {
            Ok(val) => !val.is_empty(),
            Err(_) => false,
        }
    }
}

impl IndicatorsBackend for PartialGpuBackend {
    fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) 
        -> PyResult<Py<PyArray1<f64>>> {
        // For now, delegate to CPU - add GPU kernels one by one
        crate::cpu_impls::rsi_cpu(py, prices, period)
    }
    
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) 
        -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::ema_cpu(py, prices, period)
    }
    
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize) 
        -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::sma_cpu(py, values, period)
    }
    
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) 
        -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        crate::cpu_impls::bollinger_bands_cpu(py, prices, period, std_dev)
    }
    
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, 
               close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::atr_cpu(py, high, low, close, period)
    }
    
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, 
                      close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::williams_r_cpu(py, high, low, close, period)
    }
    
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, 
               close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        crate::cpu_impls::cci_cpu(py, high, low, close, period)
    }
    
    fn vpin<'py>(&self, py: Python<'py>, buy_volumes: PyReadonlyArray1<'py, f64>,
                sell_volumes: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Py<PyArray1<f64>>> {
        let buy_vols = buy_volumes.as_array();
        let sell_vols = sell_volumes.as_array();
        let buy_slice = buy_vols.as_slice().unwrap();
        let sell_slice = sell_vols.as_slice().unwrap();
        
        let results = {
            #[cfg(feature = "cuda")]
            {
                // Use CUDA computation
                vpin_cuda_compute(buy_slice, sell_slice, window)
            }
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            {
                // Try GPU computation if CUDA is not available
                let device = WgpuDevice::default();
                let client = WgpuRuntime::client(&device);
                vpin_gpu_compute::<WgpuRuntime>(&client, buy_slice, sell_slice, window)
            }
            #[cfg(not(feature = "gpu"))]
            {
                // Fallback to CPU when no GPU features are enabled
                crate::cpu_impls::vpin_cpu_kernel(buy_slice, sell_slice, window)
            }
        };
        
        Ok(PyArray1::from_vec(py, results).to_owned().into())
    }
}
