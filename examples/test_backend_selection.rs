//! Backend Selection Validation Example
//! 
//! This example demonstrates and validates the backend selection logic
//! with different environment variable configurations.
//! 
//! Run with different environment variables:
//! - Default: `cargo run --example test_backend_selection`
//! - CPU: `RUST_INDICATORS_DEVICE=cpu cargo run --example test_backend_selection`
//! - GPU: `RUST_INDICATORS_DEVICE=gpu cargo run --example test_backend_selection`
//! - GPU with CUDA: `RUST_INDICATORS_DEVICE=gpu CUDA_VISIBLE_DEVICES=0 cargo run --example test_backend_selection`

use std::env;

fn main() {
    println!("=== Backend Selection Validation ===\n");
    
    // Display current environment
    println!("Environment Variables:");
    match env::var("RUST_INDICATORS_DEVICE") {
        Ok(device) => println!("  RUST_INDICATORS_DEVICE = {}", device),
        Err(_) => println!("  RUST_INDICATORS_DEVICE = <not set>"),
    }
    
    match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(devices) => println!("  CUDA_VISIBLE_DEVICES = {}", devices),
        Err(_) => println!("  CUDA_VISIBLE_DEVICES = <not set>"),
    }
    
    println!();
    
    // Test GPU availability check
    println!("GPU Availability Check:");
    let gpu_available = rust_indicators::backends::gpu::PartialGpuBackend::is_available();
    println!("  PartialGpuBackend::is_available() = {}", gpu_available);
    
    // Test GPU backend creation
    println!("\nGPU Backend Creation:");
    match rust_indicators::backends::gpu::PartialGpuBackend::new() {
        Ok(_) => println!("  PartialGpuBackend::new() = Ok (GPU backend created successfully)"),
        Err(e) => println!("  PartialGpuBackend::new() = Err ({})", e),
    }
    
    // Simulate the backend selection logic from RustTA::select_backend()
    println!("\nBackend Selection Logic:");
    let selected_backend = match env::var("RUST_INDICATORS_DEVICE").as_deref() {
        Ok("gpu") => {
            println!("  Environment requests GPU backend");
            match rust_indicators::backends::gpu::PartialGpuBackend::new() {
                Ok(_) => {
                    println!("  GPU backend creation successful");
                    "gpu"
                },
                Err(e) => {
                    println!("  GPU backend creation failed: {}", e);
                    println!("  Falling back to CPU backend");
                    "cpu"
                },
            }
        },
        Ok(device) => {
            println!("  Environment requests '{}' backend", device);
            println!("  Using CPU backend (default for non-'gpu' values)");
            "cpu"
        },
        Err(_) => {
            println!("  No environment variable set");
            println!("  Using CPU backend (default)");
            "cpu"
        },
    };
    
    println!("\n=== RESULT ===");
    println!("Selected Backend: {}", selected_backend);
    
    // Validation summary
    println!("\n=== VALIDATION SUMMARY ===");
    match env::var("RUST_INDICATORS_DEVICE").as_deref() {
        Ok("gpu") => {
            if gpu_available {
                if selected_backend == "gpu" {
                    println!("✓ PASS: GPU requested and available -> GPU backend selected");
                } else {
                    println!("✗ FAIL: GPU requested and available but CPU backend selected");
                }
            } else {
                if selected_backend == "cpu" {
                    println!("✓ PASS: GPU requested but unavailable -> CPU backend selected (fallback)");
                } else {
                    println!("✗ FAIL: GPU requested but unavailable yet GPU backend selected");
                }
            }
        },
        Ok(_) => {
            if selected_backend == "cpu" {
                println!("✓ PASS: Non-GPU device requested -> CPU backend selected");
            } else {
                println!("✗ FAIL: Non-GPU device requested but GPU backend selected");
            }
        },
        Err(_) => {
            if selected_backend == "cpu" {
                println!("✓ PASS: No device specified -> CPU backend selected (default)");
            } else {
                println!("✗ FAIL: No device specified but GPU backend selected");
            }
        },
    }
    
    println!("\n=== TEST SCENARIOS ===");
    println!("To test different scenarios, run:");
    println!("1. Default (CPU): cargo run --example test_backend_selection");
    println!("2. Explicit CPU: RUST_INDICATORS_DEVICE=cpu cargo run --example test_backend_selection");
    println!("3. GPU without CUDA: RUST_INDICATORS_DEVICE=gpu cargo run --example test_backend_selection");
    println!("4. GPU with CUDA: RUST_INDICATORS_DEVICE=gpu CUDA_VISIBLE_DEVICES=0 cargo run --example test_backend_selection");
    println!("5. Invalid device: RUST_INDICATORS_DEVICE=invalid cargo run --example test_backend_selection");
}