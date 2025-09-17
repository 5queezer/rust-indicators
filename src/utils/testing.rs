//! Shared test utilities for benchmarks and validation

use rand::Rng;
use std::env;
use std::time::Instant;

/// Generate test data for benchmarks
pub fn generate_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let buy_volumes: Vec<f64> = (0..size).map(|_| rng.gen_range(10.0..1000.0)).collect();
    let sell_volumes: Vec<f64> = (0..size).map(|_| rng.gen_range(10.0..1000.0)).collect();
    (buy_volumes, sell_volumes)
}

/// Benchmark a function with timing
pub fn benchmark_function<F, R>(iterations: usize, mut func: F) -> (f64, R)
where
    F: FnMut() -> R,
{
    let start = Instant::now();
    let mut result = None;

    for _ in 0..iterations {
        result = Some(func());
    }

    let duration = start.elapsed().as_secs_f64();
    (duration, result.unwrap())
}

/// Verify two result vectors match within tolerance
pub fn verify_results_match(result1: &[f64], result2: &[f64], tolerance: f64) -> bool {
    if result1.len() != result2.len() {
        return false;
    }

    result1
        .iter()
        .zip(result2.iter())
        .all(|(&val1, &val2)| (val1 - val2).abs() <= tolerance)
}

/// Calculate throughput (items per second)
pub fn calculate_throughput(size: usize, iterations: usize, time_seconds: f64) -> f64 {
    (size * iterations) as f64 / time_seconds
}

/// Format performance comparison
pub fn format_performance_comparison(name1: &str, time1: f64, name2: &str, time2: f64) -> String {
    let speedup = time1 / time2;
    if speedup > 1.0 {
        format!("{} is {:.2}x faster than {}", name2, speedup, name1)
    } else {
        format!("{} is {:.2}x faster than {}", name1, 1.0 / speedup, name2)
    }
}

// Environment variable management utilities for testing

/// Removes both RUST_INDICATORS_DEVICE and CUDA_VISIBLE_DEVICES environment variables
///
/// This is the most common cleanup pattern used across test files to ensure
/// a clean testing environment without any backend preferences.
pub fn cleanup_backend_env_vars() {
    env::remove_var("RUST_INDICATORS_DEVICE");
    env::remove_var("CUDA_VISIBLE_DEVICES");
}

/// Sets up environment for GPU testing by enabling CUDA
///
/// Sets CUDA_VISIBLE_DEVICES="0" to make GPU backend available for testing.
/// This simulates having a GPU device available.
pub fn setup_gpu_env() {
    env::set_var("CUDA_VISIBLE_DEVICES", "0");
}

/// Sets up environment for CPU-only testing
///
/// Removes CUDA_VISIBLE_DEVICES to force CPU backend usage.
/// This simulates having no GPU devices available.
pub fn setup_cpu_env() {
    env::remove_var("CUDA_VISIBLE_DEVICES");
}

/// Sets the RUST_INDICATORS_DEVICE environment variable to the specified device
///
/// # Arguments
/// * `device` - The device type to set ("cpu", "gpu", "adaptive", etc.)
///
/// # Examples
/// ```
/// use rust_indicators::utils::testing::setup_device_env;
///
/// setup_device_env("gpu");  // Request GPU backend
/// setup_device_env("cpu");  // Request CPU backend
/// ```
pub fn setup_device_env(device: &str) {
    env::set_var("RUST_INDICATORS_DEVICE", device);
}
