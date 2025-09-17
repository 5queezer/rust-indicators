use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use rust_indicators::backends::adaptive::AdaptiveBackend;
use rust_indicators::backends::cpu::CpuBackend;
use rust_indicators::core::traits::IndicatorsBackend;
use std::time::Instant;

fn benchmark_backend(backend: &dyn IndicatorsBackend, name: &str, sizes: &[usize]) {
    println!("Backend: {}", name);
    println!("Size\tTime(s)\tThroughput");

    for &size in sizes {
        let buy_volumes: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        let sell_volumes: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).cos().abs()).collect();

        let start = Instant::now();
        let iterations = if size < 5000 { 10 } else { 3 };

        for _ in 0..iterations {
            pyo3::Python::with_gil(|py| {
                let buy_array = PyArray1::from_slice(py, &buy_volumes);
                let sell_array = PyArray1::from_slice(py, &sell_volumes);
                let _ = backend.vpin(py, buy_array.readonly(), sell_array.readonly(), 50);
            });
        }

        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
        let throughput = size as f64 / elapsed;

        println!("{}\t{:.4}\t{:.0}", size, elapsed, throughput);
    }
    println!();
}

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    let test_sizes = vec![1000, 2000, 3000, 4000, 5000, 7500, 10000, 25000, 50000];

    let cpu_backend = CpuBackend::new();
    benchmark_backend(&cpu_backend, "CPU", &test_sizes);

    let adaptive_backend = AdaptiveBackend::new()?;
    benchmark_backend(&adaptive_backend, "Adaptive", &test_sizes);

    Ok(())
}
