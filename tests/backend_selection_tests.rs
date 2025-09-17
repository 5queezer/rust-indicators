use rstest::{fixture, rstest};
use rust_indicators::utils::backend_selection;
use rust_indicators::utils::testing::*;
use std::env;

#[cfg(test)]
mod backend_selection_tests {
    use super::*;
    use rust_indicators::backends::gpu::backend::PartialGpuBackend;

    // Shared fixtures for backend testing
    #[fixture]
    fn clean_environment() {
        cleanup_backend_env_vars();
    }

    #[fixture]
    fn gpu_available_environment() {
        setup_gpu_env();
    }

    #[test]
    fn test_gpu_backend_availability_check() {
        // Test GPU availability check without CUDA
        setup_cpu_env();
        assert!(
            !PartialGpuBackend::is_available(),
            "GPU should not be available without CUDA_VISIBLE_DEVICES"
        );
        println!("✓ GPU availability check without CUDA: Correctly returns false");

        // Test GPU availability check with CUDA
        setup_gpu_env();
        assert!(
            PartialGpuBackend::is_available(),
            "GPU should be available with CUDA_VISIBLE_DEVICES"
        );
        println!("✓ GPU availability check with CUDA: Correctly returns true");

        // Clean up
        setup_cpu_env();
    }

    #[test]
    fn test_gpu_backend_creation_failure() {
        // Ensure CUDA is not available
        setup_cpu_env();

        // Try to create GPU backend
        let result = PartialGpuBackend::new();

        // Should fail
        assert!(
            result.is_err(),
            "GPU backend creation should fail without CUDA"
        );
        println!("✓ GPU backend creation failure: Correctly fails when GPU unavailable");
    }

    #[test]
    fn test_gpu_backend_creation_success() {
        // Mock CUDA availability
        setup_gpu_env();

        // Try to create GPU backend
        let result = PartialGpuBackend::new();

        // Should succeed
        assert!(
            result.is_ok(),
            "GPU backend creation should succeed with CUDA"
        );
        println!("✓ GPU backend creation success: Correctly succeeds when GPU available");

        // Clean up
        setup_cpu_env();
    }

    #[rstest]
    #[case::gpu_lowercase("gpu", true)]
    #[case::gpu_uppercase("GPU", false)] // Case sensitive
    #[case::cpu("cpu", false)]
    #[case::invalid("invalid", false)]
    #[case::empty("", false)]
    fn test_environment_variable_parsing(
        #[case] env_value: &str,
        #[case] should_request_gpu: bool,
    ) {
        setup_device_env(env_value);

        let env_result = env::var("RUST_INDICATORS_DEVICE");
        let requests_gpu = matches!(env_result.as_deref(), Ok("gpu"));

        assert_eq!(
            requests_gpu,
            should_request_gpu,
            "Environment variable '{}' should {} request GPU",
            env_value,
            if should_request_gpu { "" } else { "not" }
        );

        // Clean up after each test case
        cleanup_backend_env_vars();
    }

    #[test]
    fn test_backend_selection_logic() {
        // Test the core logic that would be used in select_backend()

        // Case 1: No environment variable set
        cleanup_backend_env_vars();
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(!should_try_gpu, "Should not try GPU when no env var is set");

        // Case 2: Environment variable set to "gpu"
        setup_device_env("gpu");
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should try GPU when env var is 'gpu'");

        // Case 3: Environment variable set to "cpu"
        setup_device_env("cpu");
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(!should_try_gpu, "Should not try GPU when env var is 'cpu'");

        // Clean up
        cleanup_backend_env_vars();
        println!("✓ Backend selection logic: All environment variable cases work correctly");
    }

    #[test]
    fn test_fallback_behavior() {
        // Test the fallback logic: GPU request -> GPU unavailable -> CPU fallback
        setup_device_env("gpu");
        setup_cpu_env(); // Ensure GPU is unavailable

        // Simulate the selection logic
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should initially try GPU");

        let gpu_available = PartialGpuBackend::is_available();
        assert!(!gpu_available, "GPU should not be available");

        let gpu_creation_result = PartialGpuBackend::new();
        assert!(
            gpu_creation_result.is_err(),
            "GPU backend creation should fail"
        );

        // Use the shared backend selection logic
        let final_backend = backend_selection::select_simple_backend();

        assert_eq!(
            final_backend, "cpu",
            "Should fall back to CPU when GPU unavailable"
        );

        // Clean up
        cleanup_backend_env_vars();
        println!("✓ Fallback behavior: GPU -> CPU fallback works correctly");
    }

    #[test]
    fn test_successful_gpu_selection() {
        // Test successful GPU selection when GPU is available
        setup_device_env("gpu");
        setup_gpu_env(); // Make GPU available

        // Simulate the selection logic
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should try GPU");

        let gpu_available = PartialGpuBackend::is_available();
        assert!(gpu_available, "GPU should be available");

        let gpu_creation_result = PartialGpuBackend::new();
        assert!(
            gpu_creation_result.is_ok(),
            "GPU backend creation should succeed"
        );

        let final_backend = backend_selection::select_simple_backend();

        assert_eq!(
            final_backend, "gpu",
            "Should successfully select GPU when available"
        );

        // Clean up
        cleanup_backend_env_vars();
        println!("✓ Successful GPU selection: GPU selection works when GPU is available");
    }
}
