use crate::backends::cpu::backend::CpuBackend;
use crate::backends::gpu::partial_gpu_backend::PartialGpuBackend;
use crate::core::traits::IndicatorsBackend;
use crate::{extract_safe, gpu_method};
use anyhow::Result;
use log::{error, info, warn};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::backends::gpu::implementations::{hilbert_transform_cuda_compute, vpin_cuda_compute};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use crate::backends::gpu::implementations::{hilbert_transform_gpu_compute, vpin_gpu_compute};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

impl IndicatorsBackend for PartialGpuBackend {
    // CPU delegation methods using gpu_method! macro for non-GPU indicators
    gpu_method!(rsi, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    gpu_method!(ema, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    gpu_method!(sma, (values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    fn bollinger_bands<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f64>,
        period: usize,
        std_dev: f64,
    ) -> PyResult<crate::indicators::api::BollingerBandsOutput> {
        let result = self
            .cpu_backend
            .bollinger_bands(py, prices, period, std_dev)?;
        Ok(result)
    }
    gpu_method!(atr, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    gpu_method!(williams_r, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    gpu_method!(cci, (high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
    gpu_method!(supersmoother, (data: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);

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

        let results = if PartialGpuBackend::is_available() {
            let operation_name = "vpin";
            self.performance_monitor
                .track_operation(operation_name, || {
                    match (|| -> Result<Vec<f64>> {
                        #[cfg(feature = "cuda")]
                        {
                            info!("Attempting VPIN computation with CUDA.");
                            Ok(vpin_cuda_compute(buy_slice, sell_slice, window))
                        }
                        #[cfg(all(feature = "gpu", not(feature = "cuda")))]
                        {
                            info!("Attempting VPIN computation with WebGPU.");
                            let device = WgpuDevice::default();
                            let client = WgpuRuntime::client(&device);
                            Ok(vpin_gpu_compute::<WgpuRuntime>(
                                &client, buy_slice, sell_slice, window,
                            ))
                        }
                        #[cfg(not(any(
                            feature = "cuda",
                            all(feature = "gpu", not(feature = "cuda"))
                        )))]
                        {
                            Err(anyhow::anyhow!("GPU features not enabled at compile time."))
                        }
                    })() {
                        Ok(gpu_results) => {
                            self.gpu_failure_count
                                .store(0, std::sync::atomic::Ordering::SeqCst);
                            gpu_results
                        }
                        Err(e) => {
                            error!("GPU VPIN computation failed: {}. Falling back to CPU.", e);
                            self.gpu_failure_count
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.performance_monitor.record_fallback(operation_name);
                            self.cpu_backend._cpu_vpin(buy_slice, sell_slice, window)
                        }
                    }
                })
        } else {
            warn!("GPU not available for VPIN, falling back to CPU.");
            self.cpu_backend._cpu_vpin(buy_slice, sell_slice, window)
        };

        Ok(PyArray1::from_vec(py, results).to_owned().into())
    }

    fn hilbert_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        lp_period: usize,
    ) -> PyResult<crate::indicators::api::HilbertTransformOutput> {
        let data_array = data.as_array();
        let data_slice = extract_safe!(data_array, "data");

        let (real_vec, imag_vec) = if PartialGpuBackend::is_available() {
            let operation_name = "hilbert_transform";
            self.performance_monitor.track_operation(operation_name, || {
                match (|| -> Result<(Vec<f64>, Vec<f64>)> {
                    #[cfg(feature = "cuda")]
                    {
                        info!("Attempting Hilbert Transform computation with CUDA.");
                        Ok(hilbert_transform_cuda_compute(data_slice, lp_period))
                    }
                    #[cfg(all(feature = "gpu", not(feature = "cuda")))]
                    {
                        info!("Attempting Hilbert Transform computation with WebGPU.");
                        let device = WgpuDevice::default();
                        let client = WgpuRuntime::client(&device);
                        Ok(hilbert_transform_gpu_compute::<WgpuRuntime>(&client, data_slice, lp_period))
                    }
                    #[cfg(not(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda")))))]
                    {
                        Err(anyhow::anyhow!("GPU features not enabled at compile time."))
                    }
                })() {
                    Ok(gpu_results) => {
                        self.gpu_failure_count.store(0, std::sync::atomic::Ordering::SeqCst);
                        gpu_results
                    },
                    Err(e) => {
                        error!("GPU Hilbert Transform computation failed: {}. Falling back to CPU.", e);
                        self.gpu_failure_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        self.performance_monitor.record_fallback(operation_name);
                        self.cpu_backend._cpu_hilbert_transform(data_slice, lp_period)
                    },
                }
            })
        } else {
            warn!("GPU not available for Hilbert Transform, falling back to CPU.");
            self.cpu_backend
                ._cpu_hilbert_transform(data_slice, lp_period)
        };

        Ok(crate::indicators::api::HilbertTransformOutput {
            real: PyArray1::from_vec(py, real_vec).to_owned().into(),
            imag: PyArray1::from_vec(py, imag_vec).to_owned().into(),
        })
    }
}
