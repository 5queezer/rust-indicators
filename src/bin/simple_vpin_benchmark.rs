use std::time::Instant;
use rand::Rng;
use rust_indicators::cpu_impls::vpin_cpu_kernel;

#[cfg(feature = "cuda")]
use rust_indicators::gpu_impls::vpin_cuda_compute;

#[cfg(feature = "cuda")]
fn generate_test_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let buy_volumes: Vec<f64> = (0..size).map(|_| rng.gen_range(10.0..1000.0)).collect();
    let sell_volumes: Vec<f64> = (0..size).map(|_| rng.gen_range(10.0..1000.0)).collect();
    (buy_volumes, sell_volumes)
}

#[cfg(feature = "cuda")]
fn benchmark_cpu(buy_volumes: &[f64], sell_volumes: &[f64], window: usize, iterations: usize) -> (f64, Vec<f64>) {
    let start = Instant::now();
    let mut result = Vec::new();
    
    for _ in 0..iterations {
        result = vpin_cpu_kernel(buy_volumes, sell_volumes, window);
    }
    
    let duration = start.elapsed().as_secs_f64();
    (duration, result)
}

#[cfg(feature = "cuda")]
fn benchmark_cuda(buy_volumes: &[f64], sell_volumes: &[f64], window: usize, iterations: usize) -> Result<(f64, Vec<f64>), String> {
    let start = Instant::now();
    let mut result = Vec::new();
    
    for _ in 0..iterations {
        // Wrap in a panic catch to handle CUDA errors gracefully
        result = match std::panic::catch_unwind(|| {
            vpin_cuda_compute(buy_volumes, sell_volumes, window)
        }) {
            Ok(res) => res,
            Err(_) => return Err("CUDA computation failed".to_string()),
        };
    }
    
    let duration = start.elapsed().as_secs_f64();
    Ok((duration, result))
}

#[cfg(not(feature = "cuda"))]
fn benchmark_cuda(_buy_volumes: &[f64], _sell_volumes: &[f64], _window: usize, _iterations: usize) -> Result<(f64, Vec<f64>), String> {
    Err("CUDA feature not enabled. Please run with --features cuda".to_string())
}

#[cfg(feature = "cuda")]
fn verify_results_match(cpu_result: &[f64], cuda_result: &[f64], tolerance: f64) -> bool {
    if cpu_result.len() != cuda_result.len() {
        return false;
    }
    
    for (i, (&cpu_val, &cuda_val)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
        let diff = (cpu_val - cuda_val).abs();
        if diff > tolerance {
            println!("Mismatch at index {}: CPU={:.6}, CUDA={:.6}, diff={:.6}", i, cpu_val, cuda_val, diff);
            return false;
        }
    }
    true
}

fn main() {
    println!("VPIN Benchmark: CPU vs CUDA");
    println!("============================");
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: CUDA feature not enabled!");
        println!("Please run with: cargo run --bin simple_vpin_benchmark --features cuda");
        return;
    }
    
    #[cfg(feature = "cuda")]
    {
        println!("CUDA feature enabled - running real CUDA benchmarks");
        println!();
        
        let test_sizes = vec![1_000, 10_000, 50_000]; // Reduced max size to avoid CUDA errors
        let window_sizes = vec![10, 50, 200];
        let iterations = 5;
        let tolerance = 1e-6; // Tolerance for floating point comparison
        
        for &size in &test_sizes {
            println!("Dataset size: {} points", size);
            let (buy_volumes, sell_volumes) = generate_test_data(size);
            
            for &window in &window_sizes {
                println!("\n  Window size: {}", window);
                
                // CPU Benchmark
                let (cpu_time, cpu_result) = benchmark_cpu(&buy_volumes, &sell_volumes, window, iterations);
                let cpu_throughput = (size * iterations) as f64 / cpu_time;
                println!("    CPU:  {:.4}s ({:.0} points/sec)", cpu_time, cpu_throughput);
                
                // CUDA Benchmark
                match benchmark_cuda(&buy_volumes, &sell_volumes, window, iterations) {
                    Ok((cuda_time, cuda_result)) => {
                        let cuda_throughput = (size * iterations) as f64 / cuda_time;
                        println!("    CUDA: {:.4}s ({:.0} points/sec)", cuda_time, cuda_throughput);
                        
                        // Verify results are identical (within tolerance)
                        let results_match = verify_results_match(&cpu_result, &cuda_result, tolerance);
                        println!("    Results match: {}", results_match);
                        
                        if !results_match {
                            println!("    WARNING: Results don't match within tolerance {:.2e}", tolerance);
                            // Show first few values for debugging
                            println!("    First 5 CPU values:  {:?}", &cpu_result[..5.min(cpu_result.len())]);
                            println!("    First 5 CUDA values: {:?}", &cuda_result[..5.min(cuda_result.len())]);
                        }
                        
                        // Show speedup ratio
                        let cuda_speedup = cpu_time / cuda_time;
                        println!("    CUDA speedup: {:.2}x", cuda_speedup);
                        
                        // Performance analysis
                        if cuda_speedup > 1.0 {
                            println!("    Performance: CUDA is {:.1}x faster", cuda_speedup);
                        } else {
                            println!("    Performance: CPU is {:.1}x faster (CUDA overhead)", 1.0 / cuda_speedup);
                        }
                    }
                    Err(error) => {
                        println!("    CUDA: FAILED - {}", error);
                        println!("    Skipping CUDA benchmark for this configuration");
                    }
                }
                
                // Memory usage estimate
                let memory_mb = (size * 3 * 8) as f64 / 1_000_000.0; // 3 arrays * 8 bytes per f64
                println!("    Memory usage: {:.1} MB", memory_mb);
            }
            println!();
        }
        
        println!("============================");
        println!("PERFORMANCE SUMMARY:");
        println!("• CPU: Consistent performance, good for small datasets");
        println!("• CUDA: Better for large datasets, has initialization overhead");
        println!("• CUDA speedup depends on dataset size and GPU capabilities");
        println!("• Memory transfer costs can impact performance for smaller datasets");
        println!("• Results should match within floating-point precision limits");
    }
}