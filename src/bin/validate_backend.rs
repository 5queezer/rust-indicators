//! Backend Selection Validation Binary
//!
//! This binary validates the backend selection logic by testing the actual
//! PartialGpuBackend availability checks that RustTA uses internally.

use rust_indicators::utils::backend_selection;
use rust_indicators::utils::testing::*;

fn test_backend_selection_logic(
    device_env: Option<&str>,
    cuda_env: Option<&str>,
) -> (&'static str, String) {
    // Set up environment
    match device_env {
        Some(val) => setup_device_env(val),
        None => cleanup_backend_env_vars(),
    }

    match cuda_env {
        Some(_) => setup_gpu_env(),
        None => setup_cpu_env(),
    }

    // Use the shared backend selection logic
    let selected_backend = backend_selection::select_simple_backend();

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
    cleanup_backend_env_vars();

    println!("=== ALL TESTS PASSED ===");
    println!("Backend selection logic matches RustTA implementation!");
}
