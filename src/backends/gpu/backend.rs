// backend_gpu.rs - Keep it stupid simple
use crate::core::traits::IndicatorsBackend;
use crate::backends::cpu::backend::CpuBackend;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::Runtime;
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use crate::backends::gpu::implementations::vpin_gpu_compute;
#[cfg(feature = "cuda")]
use crate::backends::gpu::implementations::{vpin_cuda_compute, hilbert_transform_cuda_compute};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use crate::backends::gpu::implementations::hilbert_transform_gpu_compute;

/// Partial GPU Backend - Only VPIN uses GPU acceleration, all other methods fall back to CPU
/// This is an honest representation of the current implementation state.
pub struct PartialGpuBackend {
    cpu_backend: CpuBackend,
}

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
                    return Ok(PartialGpuBackend {
                        cpu_backend: CpuBackend::new(),
                    });
                }
            }
        }
        
        // In production, we could add more sophisticated GPU initialization here
        // For now, if is_available() returns true, we assume GPU is ready
        Ok(PartialGpuBackend {
            cpu_backend: CpuBackend::new(),
        })
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
        // Delegate to CPU backend
        self.cpu_backend.rsi(py, prices, period)
    }
    
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>> {
        self.cpu_backend.ema(py, prices, period)
    }
    
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>> {
        self.cpu_backend.sma(py, values, period)
    }
    
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64)
        -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        self.cpu_backend.bollinger_bands(py, prices, period, std_dev)
    }
    
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>,
               close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.cpu_backend.atr(py, high, low, close, period)
    }
    
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>,
                      close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.cpu_backend.williams_r(py, high, low, close, period)
    }
    
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>,
               close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        self.cpu_backend.cci(py, high, low, close, period)
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
                crate::utils::benchmarking::benchmark_vpin_cpu(buy_slice, sell_slice, window)
            }
        };
        
        Ok(PyArray1::from_vec(py, results).to_owned().into())
    }
    
    fn supersmoother<'py>(&self, py: Python<'py>, data: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>> {
        // Delegate to CPU backend - SuperSmoother has sequential dependencies
        self.cpu_backend.supersmoother(py, data, period)
    }
    
    fn hilbert_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray1<'py, f64>, lp_period: usize)
        -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        // For now, delegate to CPU backend since GPU implementation has complex dependencies
        // TODO: Implement proper GPU acceleration for parallelizable parts
        self.cpu_backend.hilbert_transform(py, data, lp_period)
    }
}
