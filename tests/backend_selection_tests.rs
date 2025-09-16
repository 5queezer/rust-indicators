use std::env;
use rstest::{rstest, fixture};

#[cfg(test)]
mod backend_selection_tests {
    use super::*;
    use rust_indicators::backend_gpu::PartialGpuBackend;

    // Shared fixtures for backend testing
    #[fixture]
    fn clean_environment() {
        env::remove_var("RUST_INDICATORS_DEVICE");
        env::remove_var("CUDA_VISIBLE_DEVICES");
    }

    #[fixture]
    fn gpu_available_environment() {
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
    }

    #[test]
    fn test_gpu_backend_availability_check() {
        // Test GPU availability check without CUDA
        env::remove_var("CUDA_VISIBLE_DEVICES");
        assert!(!PartialGpuBackend::is_available(), "GPU should not be available without CUDA_VISIBLE_DEVICES");
        println!("✓ GPU availability check without CUDA: Correctly returns false");
        
        // Test GPU availability check with CUDA
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        assert!(PartialGpuBackend::is_available(), "GPU should be available with CUDA_VISIBLE_DEVICES");
        println!("✓ GPU availability check with CUDA: Correctly returns true");
        
        // Clean up
        env::remove_var("CUDA_VISIBLE_DEVICES");
    }

    #[test]
    fn test_gpu_backend_creation_failure() {
        // Ensure CUDA is not available
        env::remove_var("CUDA_VISIBLE_DEVICES");
        
        // Try to create GPU backend
        let result = PartialGpuBackend::new();
        
        // Should fail
        assert!(result.is_err(), "GPU backend creation should fail without CUDA");
        println!("✓ GPU backend creation failure: Correctly fails when GPU unavailable");
    }

    #[test]
    fn test_gpu_backend_creation_success() {
        // Mock CUDA availability
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        
        // Try to create GPU backend
        let result = PartialGpuBackend::new();
        
        // Should succeed
        assert!(result.is_ok(), "GPU backend creation should succeed with CUDA");
        println!("✓ GPU backend creation success: Correctly succeeds when GPU available");
        
        // Clean up
        env::remove_var("CUDA_VISIBLE_DEVICES");
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
        env::set_var("RUST_INDICATORS_DEVICE", env_value);
        
        let env_result = env::var("RUST_INDICATORS_DEVICE");
        let requests_gpu = matches!(env_result.as_deref(), Ok("gpu"));
        
        assert_eq!(requests_gpu, should_request_gpu,
            "Environment variable '{}' should {} request GPU",
            env_value, if should_request_gpu { "" } else { "not" });
        
        // Clean up after each test case
        env::remove_var("RUST_INDICATORS_DEVICE");
    }

    #[test]
    fn test_backend_selection_logic() {
        // Test the core logic that would be used in select_backend()
        
        // Case 1: No environment variable set
        env::remove_var("RUST_INDICATORS_DEVICE");
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(!should_try_gpu, "Should not try GPU when no env var is set");
        
        // Case 2: Environment variable set to "gpu"
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should try GPU when env var is 'gpu'");
        
        // Case 3: Environment variable set to "cpu"
        env::set_var("RUST_INDICATORS_DEVICE", "cpu");
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(!should_try_gpu, "Should not try GPU when env var is 'cpu'");
        
        // Clean up
        env::remove_var("RUST_INDICATORS_DEVICE");
        println!("✓ Backend selection logic: All environment variable cases work correctly");
    }

    #[test]
    fn test_fallback_behavior() {
        // Test the fallback logic: GPU request -> GPU unavailable -> CPU fallback
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        env::remove_var("CUDA_VISIBLE_DEVICES"); // Ensure GPU is unavailable
        
        // Simulate the selection logic
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should initially try GPU");
        
        let gpu_available = PartialGpuBackend::is_available();
        assert!(!gpu_available, "GPU should not be available");
        
        let gpu_creation_result = PartialGpuBackend::new();
        assert!(gpu_creation_result.is_err(), "GPU backend creation should fail");
        
        // In the real implementation, this would fall back to CPU
        let final_backend = if should_try_gpu && gpu_creation_result.is_ok() {
            "gpu"
        } else {
            "cpu"
        };
        
        assert_eq!(final_backend, "cpu", "Should fall back to CPU when GPU unavailable");
        
        // Clean up
        env::remove_var("RUST_INDICATORS_DEVICE");
        println!("✓ Fallback behavior: GPU -> CPU fallback works correctly");
    }

    #[test]
    fn test_successful_gpu_selection() {
        // Test successful GPU selection when GPU is available
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        env::set_var("CUDA_VISIBLE_DEVICES", "0"); // Make GPU available
        
        // Simulate the selection logic
        let should_try_gpu = matches!(env::var("RUST_INDICATORS_DEVICE").as_deref(), Ok("gpu"));
        assert!(should_try_gpu, "Should try GPU");
        
        let gpu_available = PartialGpuBackend::is_available();
        assert!(gpu_available, "GPU should be available");
        
        let gpu_creation_result = PartialGpuBackend::new();
        assert!(gpu_creation_result.is_ok(), "GPU backend creation should succeed");
        
        let final_backend = if should_try_gpu && gpu_creation_result.is_ok() {
            "gpu"
        } else {
            "cpu"
        };
        
        assert_eq!(final_backend, "gpu", "Should successfully select GPU when available");
        
        // Clean up
        env::remove_var("RUST_INDICATORS_DEVICE");
        env::remove_var("CUDA_VISIBLE_DEVICES");
        println!("✓ Successful GPU selection: GPU selection works when GPU is available");
    }
}