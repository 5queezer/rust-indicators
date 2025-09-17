//! Backend Selection Utilities
//!
//! This module provides centralized backend selection logic to eliminate
//! code duplication across the codebase. It handles GPU availability checking,
//! environment variable parsing, and backend selection with proper fallback.

use crate::backends::adaptive::AdaptiveBackend;
use crate::backends::cpu::CpuBackend;
use crate::backends::gpu::PartialGpuBackend;
use crate::core::traits::IndicatorsBackend;
use pyo3::prelude::*;
use std::env;

/// Result type for backend selection operations
pub type BackendSelectionResult = (Box<dyn IndicatorsBackend>, &'static str);

/// Checks if GPU backend is available by verifying CUDA environment
pub fn is_gpu_available() -> bool {
    PartialGpuBackend::is_available()
}

/// Attempts to create a GPU backend, returns Ok if successful
pub fn try_create_gpu_backend() -> PyResult<PartialGpuBackend> {
    PartialGpuBackend::new()
}

/// Gets the requested device from environment variable
pub fn get_requested_device() -> Option<String> {
    env::var("RUST_INDICATORS_DEVICE").ok()
}

/// Selects the appropriate backend based on environment configuration
///
/// This function replicates the exact logic from RustTA::select_backend()
/// with proper fallback handling:
/// - "cpu" -> CPU backend
/// - "gpu" -> GPU backend (falls back to CPU if GPU unavailable)  
/// - "adaptive" -> Adaptive backend (falls back to CPU if unavailable)
/// - default -> Adaptive backend (falls back to CPU if unavailable)
pub fn select_backend() -> BackendSelectionResult {
    match get_requested_device().as_deref() {
        Some("cpu") => (Box::new(CpuBackend::new()), "cpu"),
        Some("gpu") => match try_create_gpu_backend() {
            Ok(backend) => (Box::new(backend), "gpu"),
            Err(_) => (Box::new(CpuBackend::new()), "cpu"),
        },
        Some("adaptive") => match AdaptiveBackend::new() {
            Ok(backend) => (Box::new(backend), "adaptive"),
            Err(_) => (Box::new(CpuBackend::new()), "cpu"),
        },
        _ => match AdaptiveBackend::new() {
            Ok(backend) => (Box::new(backend), "adaptive"),
            Err(_) => (Box::new(CpuBackend::new()), "cpu"),
        },
    }
}

/// Simplified backend selection for cases that only need CPU/GPU choice
///
/// This function handles the common pattern found in validation and test code:
/// - If "gpu" is requested and available -> "gpu"
/// - Otherwise -> "cpu"
///
/// # Environment Variables
///
/// The function respects the following environment variables:
/// - `RUST_INDICATORS_DEVICE`: Set to "gpu" to request GPU backend, any other value defaults to CPU
/// - `CUDA_VISIBLE_DEVICES`: Must be set for GPU backend to be considered available
///
/// # Examples
///
/// ## Default behavior (no environment variables)
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// // Clean environment
/// env::remove_var("RUST_INDICATORS_DEVICE");
/// env::remove_var("CUDA_VISIBLE_DEVICES");
///
/// let backend = select_simple_backend();
/// assert_eq!(backend, "cpu"); // Default to CPU when no GPU requested
/// ```
///
/// ## Explicit CPU request
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// env::set_var("RUST_INDICATORS_DEVICE", "cpu");
/// env::remove_var("CUDA_VISIBLE_DEVICES");
///
/// let backend = select_simple_backend();
/// assert_eq!(backend, "cpu");
///
/// // Cleanup
/// env::remove_var("RUST_INDICATORS_DEVICE");
/// ```
///
/// ## GPU request without CUDA (fallback to CPU)
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// env::set_var("RUST_INDICATORS_DEVICE", "gpu");
/// env::remove_var("CUDA_VISIBLE_DEVICES"); // No CUDA available
///
/// let backend = select_simple_backend();
/// assert_eq!(backend, "cpu"); // Falls back to CPU when GPU unavailable
///
/// // Cleanup
/// env::remove_var("RUST_INDICATORS_DEVICE");
/// ```
///
/// ## GPU request with CUDA available
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// env::set_var("RUST_INDICATORS_DEVICE", "gpu");
/// env::set_var("CUDA_VISIBLE_DEVICES", "0"); // CUDA device available
///
/// let backend = select_simple_backend();
/// assert_eq!(backend, "gpu"); // GPU backend selected when available
///
/// // Cleanup
/// env::remove_var("RUST_INDICATORS_DEVICE");
/// env::remove_var("CUDA_VISIBLE_DEVICES");
/// ```
///
/// ## Invalid device specification
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// env::set_var("RUST_INDICATORS_DEVICE", "invalid");
///
/// let backend = select_simple_backend();
/// assert_eq!(backend, "cpu"); // Non-"gpu" values default to CPU
///
/// // Cleanup
/// env::remove_var("RUST_INDICATORS_DEVICE");
/// ```
///
/// # Error Handling
///
/// This function never panics and always returns a valid backend choice.
/// If GPU backend creation fails for any reason (missing CUDA, initialization errors, etc.),
/// it gracefully falls back to the CPU backend.
///
/// # Usage in Validation Code
///
/// This function is commonly used in validation and testing scenarios where you need
/// to determine which backend was actually selected based on environment configuration:
///
/// ```rust
/// use std::env;
/// use rust_indicators::utils::backend_selection::select_simple_backend;
///
/// // Display current configuration
/// match env::var("RUST_INDICATORS_DEVICE") {
///     Ok(device) => println!("Requested device: {}", device),
///     Err(_) => println!("No device specified (default: CPU)"),
/// }
///
/// let selected = select_simple_backend();
/// println!("Selected backend: {}", selected);
///
/// // Validate selection logic
/// match env::var("RUST_INDICATORS_DEVICE").as_deref() {
///     Ok("gpu") => {
///         if selected == "gpu" {
///             println!("✓ GPU backend successfully created");
///         } else {
///             println!("⚠ GPU requested but fell back to CPU (GPU unavailable)");
///         }
///     },
///     _ => {
///         assert_eq!(selected, "cpu");
///         println!("✓ Using CPU backend as expected");
///     }
/// }
/// ```
pub fn select_simple_backend() -> &'static str {
    match get_requested_device().as_deref() {
        Some("gpu") => match try_create_gpu_backend() {
            Ok(_) => "gpu",
            Err(_) => "cpu",
        },
        _ => "cpu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_gpu_availability_check() {
        // Test without CUDA
        env::remove_var("CUDA_VISIBLE_DEVICES");
        assert!(!is_gpu_available());

        // Test with CUDA
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        assert!(is_gpu_available());

        // Clean up
        env::remove_var("CUDA_VISIBLE_DEVICES");
    }

    #[test]
    fn test_device_request_parsing() {
        // Test no environment variable
        env::remove_var("RUST_INDICATORS_DEVICE");
        assert_eq!(get_requested_device(), None);

        // Test GPU request
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        assert_eq!(get_requested_device(), Some("gpu".to_string()));

        // Test CPU request
        env::set_var("RUST_INDICATORS_DEVICE", "cpu");
        assert_eq!(get_requested_device(), Some("cpu".to_string()));

        // Clean up
        env::remove_var("RUST_INDICATORS_DEVICE");
    }

    #[test]
    fn test_simple_backend_selection() {
        // Clean up any existing environment variables first
        env::remove_var("RUST_INDICATORS_DEVICE");
        env::remove_var("CUDA_VISIBLE_DEVICES");

        // Test default (no env var)
        assert_eq!(select_simple_backend(), "cpu");

        // Test explicit CPU
        env::set_var("RUST_INDICATORS_DEVICE", "cpu");
        assert_eq!(select_simple_backend(), "cpu");

        // Test GPU without CUDA (should fallback to CPU)
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        env::remove_var("CUDA_VISIBLE_DEVICES");
        assert_eq!(select_simple_backend(), "cpu");

        // Test GPU with CUDA
        env::set_var("RUST_INDICATORS_DEVICE", "gpu");
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        assert_eq!(select_simple_backend(), "gpu");

        // Clean up
        env::remove_var("RUST_INDICATORS_DEVICE");
        env::remove_var("CUDA_VISIBLE_DEVICES");
    }
}
