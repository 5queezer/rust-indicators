#[cfg(feature = "cuda")]
use rust_indicators::backends::cpu::implementations::vpin_cpu_kernel;
#[cfg(feature = "cuda")]
use rust_indicators::backends::gpu::implementations::vpin_cuda_compute;
#[cfg(feature = "cuda")]
use rust_indicators::utils::testing::*;

#[cfg(feature = "cuda")]
fn find_crossover_point() {
    println!("Finding VPIN CPU/GPU crossover point...");
    println!("Size\tCPU(s)\tGPU(s)\tWinner\tRatio");

    let window = 50;
    let iterations = 5;
    let mut crossover_found = false;
    let mut crossover_point = 0;

    for size in (1000..=10000).step_by(500) {
        let (buy_volumes, sell_volumes) = generate_test_data(size);

        let (cpu_time, _) = benchmark_function(iterations, || {
            vpin_cpu_kernel(&buy_volumes, &sell_volumes, window)
        });

        let (gpu_time, _) = benchmark_function(iterations, || {
            vpin_cuda_compute(&buy_volumes, &sell_volumes, window)
        });

        let winner = if cpu_time < gpu_time { "CPU" } else { "GPU" };
        let ratio = if cpu_time < gpu_time {
            gpu_time / cpu_time
        } else {
            cpu_time / gpu_time
        };

        println!(
            "{}\t{:.4}\t{:.4}\t{}\t{:.2}x",
            size, cpu_time, gpu_time, winner, ratio
        );

        if !crossover_found && gpu_time < cpu_time {
            crossover_point = size;
            crossover_found = true;
        }
    }

    if crossover_found {
        println!(
            "\nCrossover point found at: {} data points",
            crossover_point
        );
    } else {
        println!("\nNo crossover point found in tested range");
    }
}

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: CUDA feature not enabled!");
        println!("Run with: cargo run --bin find_vpin_crossover --features cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    find_crossover_point();
}
