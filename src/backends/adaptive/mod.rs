use crate::core::traits::IndicatorsBackend;
use crate::backends::cpu::CpuBackend;
use crate::backends::gpu::PartialGpuBackend;
use crate::delegate_indicator;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, Duration, Instant};

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
        }
    }
}

pub struct PerformanceProfile {
    pub thresholds: HashMap<String, usize>,
    pub last_calibration: SystemTime,
    pub calibration_interval: Duration,
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
        
        Self {
            thresholds,
            last_calibration: SystemTime::now(),
            calibration_interval: Duration::from_secs(3600),
        }
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
        
        let threshold = self.performance_profile.thresholds
            .get(indicator)
            .copied()
            .unwrap_or(usize::MAX);
            
        params.computational_complexity() >= threshold
    }
    
    pub fn calibrate_performance(&mut self, indicator: &str) -> PyResult<()> {
        if !self.gpu_available {
            return Ok(());
        }
        
        match indicator {
            "vpin" => self.calibrate_vpin(),
            _ => Ok(()), // Other indicators use CPU-only for now
        }
    }
    
    fn calibrate_vpin(&mut self) -> PyResult<()> {
        let test_sizes = vec![2000, 3000, 4000, 5000, 6000, 7000];
        let window = 50;
        let iterations = 3;
        
        for &size in &test_sizes {
            let (cpu_time, gpu_time) = self.benchmark_vpin_at_size(size, window, iterations)?;
            
            if gpu_time < cpu_time {
                self.performance_profile.thresholds.insert("vpin".to_string(), size);
                break;
            }
        }
        
        self.performance_profile.last_calibration = SystemTime::now();
        Ok(())
    }
    
    fn benchmark_vpin_at_size(&self, size: usize, window: usize, iterations: usize) -> PyResult<(f64, f64)> {
        let buy_volumes: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        let sell_volumes: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).cos().abs()).collect();
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::utils::benchmarking::benchmark_vpin_cpu(&buy_volumes, &sell_volumes, window);
        }
        let cpu_time = start.elapsed().as_secs_f64() / iterations as f64;
        
        let gpu_time = if self.gpu_backend.is_some() {
            let start = Instant::now();
            for _ in 0..iterations {
                #[cfg(feature = "cuda")]
                {
                    let _ = crate::backends::gpu::implementations::vpin_cuda_compute(&buy_volumes, &sell_volumes, window);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = crate::utils::benchmarking::benchmark_vpin_cpu(&buy_volumes, &sell_volumes, window);
                }
            }
            start.elapsed().as_secs_f64() / iterations as f64
        } else {
            f64::MAX
        };
        
        Ok((cpu_time, gpu_time))
    }
    
    fn needs_recalibration(&self) -> bool {
        self.performance_profile.last_calibration.elapsed()
            .unwrap_or(Duration::MAX) > self.performance_profile.calibration_interval
    }
}

impl IndicatorsBackend for AdaptiveBackend {
    fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "rsi",
            IndicatorParams::Rsi { data_size: prices.as_array().len(), period },
            rsi(prices, period)
        )
    }
    
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "ema",
            IndicatorParams::Ema { data_size: prices.as_array().len(), period },
            ema(prices, period)
        )
    }
    
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "sma",
            IndicatorParams::Sma { data_size: values.as_array().len(), period },
            sma(values, period)
        )
    }
    
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        delegate_indicator!(
            self, py, "bollinger_bands",
            IndicatorParams::BollingerBands { data_size: prices.as_array().len(), period },
            bollinger_bands(prices, period, std_dev)
        )
    }
    
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "atr",
            IndicatorParams::Atr { data_size: high.as_array().len(), period },
            atr(high, low, close, period)
        )
    }
    
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "williams_r",
            IndicatorParams::WilliamsR { data_size: high.as_array().len(), period },
            williams_r(high, low, close, period)
        )
    }
    
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "cci",
            IndicatorParams::Cci { data_size: high.as_array().len(), period },
            cci(high, low, close, period)
        )
    }
    
    fn vpin<'py>(&self, py: Python<'py>, buy_volumes: PyReadonlyArray1<'py, f64>, sell_volumes: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Py<PyArray1<f64>>> {
        delegate_indicator!(
            self, py, "vpin",
            IndicatorParams::Vpin { data_size: buy_volumes.as_array().len(), window },
            vpin(buy_volumes, sell_volumes, window)
        )
    }
}