#[cfg(feature = "cuda")]
use rust_indicators::cpu_impls::vpin_cpu_kernel;
#[cfg(feature = "cuda")]
use rust_indicators::test_utils::*;
#[cfg(feature = "cuda")]
use rust_indicators::gpu_impls::vpin_cuda_compute;

#[cfg(feature = "cuda")]
fn benchmark_cuda(buy_volumes: &[f64], sell_volumes: &[f64], window: usize, iterations: usize) -> Result<(f64, Vec<f64>), String> {
    match std::panic::catch_unwind(|| {
        benchmark_function(iterations, || {
            vpin_cuda_compute(buy_volumes, sell_volumes, window)
        })
    }) {
        Ok(result) => Ok(result),
        Err(_) => Err("CUDA computation failed".to_string()),
    }
}

#[cfg(not(feature = "cuda"))]
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn benchmark_cuda(_: &[f64], _: &[f64], _: usize, _: usize) -> Result<(f64, Vec<f64>), String> {
    Err("CUDA feature not enabled".to_string())
}

#[cfg(feature = "cuda")]
fn run_benchmark(size: usize, window: usize, iterations: usize) {
    let (buy_volumes, sell_volumes) = generate_test_data(size);
    
    // CPU benchmark
    let (cpu_time, cpu_result) = benchmark_function(iterations, || {
        vpin_cpu_kernel(&buy_volumes, &sell_volumes, window)
    });
    
    let cpu_throughput = calculate_throughput(size, iterations, cpu_time);
    println!("  CPU:  {:.4}s ({:.0} points/sec)", cpu_time, cpu_throughput);
    
    // CUDA benchmark
    match benchmark_cuda(&buy_volumes, &sell_volumes, window, iterations) {
        Ok((cuda_time, cuda_result)) => {
            let cuda_throughput = calculate_throughput(size, iterations, cuda_time);
            println!("  CUDA: {:.4}s ({:.0} points/sec)", cuda_time, cuda_throughput);
            
            let results_match = verify_results_match(&cpu_result, &cuda_result, 1e-6);
            println!("  Results match: {}", results_match);
            println!("  {}", format_performance_comparison("CPU", cpu_time, "CUDA", cuda_time));
        }
        Err(error) => {
            println!("  CUDA: FAILED - {}", error);
        }
    }
}

fn main() {
    println!("VPIN Benchmark: CPU vs CUDA");
    println!("============================");
    
    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: CUDA feature not enabled!");
        println!("Run with: cargo run --bin simple_vpin_benchmark --features cuda");
        return;
    }
    
    #[cfg(feature = "cuda")]
    {
        let test_configs = [
            (1_000, 10),
            (10_000, 50),
            (50_000, 200),
        ];
        let iterations = 5;
        
        for &(size, window) in &test_configs {
            println!("\nDataset: {} points, Window: {}", size, window);
            run_benchmark(size, window, iterations);
        }
    }
}