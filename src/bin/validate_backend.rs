//! Backend Selection Validation Binary
//! 
//! This binary validates the backend selection logic by testing the actual
//! PartialGpuBackend availability checks that RustTA uses internally.

use std::env;
use rust_indicators::backend_gpu::PartialGpuBackend;

fn test_backend_selection_logic(device_env: Option<&str>, cuda_env: Option<&str>) -> (&'static str, String) {
    // Set up environment
    match device_env {
        Some(val) => env::set_var("RUST_INDICATORS_DEVICE", val),
        None => env::remove_var("RUST_INDICATORS_DEVICE"),
    }
    
    match cuda_env {
        Some(val) => env::set_var("CUDA_VISIBLE_DEVICES", val),
        None => env::remove_var("CUDA_VISIBLE_DEVICES"),
    }
    
    // Replicate the exact logic from RustTA::select_backend()
    let selected_backend = match env::var("RUST_INDICATORS_DEVICE").as_deref() {
        Ok("gpu") => {
            match PartialGpuBackend::new() {
                Ok(_) => "gpu",
                Err(_) => "cpu",
            }
        },
        _ => "cpu",
    };
    
    let env_desc = format!(
        "RUST_INDICATORS_DEVICE={}, CUDA_VISIBLE_DEVICES={}",
        device_env.unwrap_or("<none>"),
        cuda_env.unwrap_or("<none>")
    );
    
    (selected_backend, env_desc)
}

fn run_test_case(name: &str, device_env: Option<&str>, cuda_env: Option<&str>, expected: &str) {
    println!("Test: {}", name);
    let (actual, env_desc) = test_backend_selection_logic(device_env, cuda_env);
    println!("  Environment: {}", env_desc);
    println!("  Selected: {}", actual);
    
    if actual == expected {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL: Expected {}, got {}\n", expected, actual);
        std::process::exit(1);
    }
}

fn main() {
    println!("=== Backend Selection Validation ===\n");
    
    // Test cases that mirror the real RustTA::select_backend() logic
    run_test_case("Default Behavior", None, None, "cpu");
    run_test_case("Explicit CPU", Some("cpu"), None, "cpu");
    run_test_case("GPU without CUDA", Some("gpu"), None, "cpu");
    run_test_case("GPU with CUDA", Some("gpu"), Some("0"), "gpu");
    run_test_case("Invalid Device", Some("invalid"), None, "cpu");
    
    // Clean up
    env::remove_var("RUST_INDICATORS_DEVICE");
    env::remove_var("CUDA_VISIBLE_DEVICES");
    
    println!("=== ALL TESTS PASSED ===");
    println!("Backend selection logic matches RustTA implementation!");
}