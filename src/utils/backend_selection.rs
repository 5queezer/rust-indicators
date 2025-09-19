//! Backend Selection Utilities
//!
//! This module provides centralized backend selection logic to eliminate
//! code duplication across the codebase. It handles GPU availability checking,
//! environment variable parsing, and backend selection with proper fallback.

use crate::backends::adaptive::AdaptiveBackend;
use crate::backends::cpu::backend::CpuBackend;
use crate::backends::cpu::ml_backend::CpuMLBackend; // New import
use crate::backends::gpu::PartialGpuBackend;
use crate::config::gpu_config::GpuConfig;
use crate::core::traits::IndicatorsBackend;
use crate::ml::traits::MLBackend;
use crate::utils::gpu_validation::{detect_gpus, validate_gpu_config};
use log::{debug, error, info, warn};
use pyo3::prelude::*;
use std::env;
use std::sync::Arc; // Import Arc

/// Result type for backend selection operations
pub type BackendSelectionResult = (Box<dyn IndicatorsBackend>, &'static str);

/// Checks if GPU backend is available by verifying CUDA environment
/// Checks if GPU backend is available by verifying CUDA environment and configuration.
pub fn is_gpu_available() -> bool {
    let config = GpuConfig::get();
    let gpus = detect_gpus();
    validate_gpu_config(&config, &gpus) && PartialGpuBackend::is_available()
}

/// Attempts to create a GPU backend, returns Ok if successful.
/// This function now considers the GpuConfig for validation.
pub fn try_create_gpu_backend() -> Result<PartialGpuBackend> {
    let config = GpuConfig::get();

    if !config.enabled {
        return Err(anyhow::anyhow!("GPU backend is disabled by configuration."));
    }

    // Perform more robust GPU capability detection here if needed.
    // For now, we rely on PartialGpuBackend::new() to do the heavy lifting
    // and return an error if the GPU is not suitable or available.
    match PartialGpuBackend::new() {
        Ok(backend) => Ok(backend),
        Err(e) => Err(anyhow::anyhow!("Failed to create GPU backend: {:?}", e)),
    }
}

// Placeholder for a function that would retrieve detailed GPU information
// In a real application, this would interact with CUDA APIs (e.g., `cudaGetDeviceProperties`).
#[derive(Debug, Default)]
pub struct GpuInfo {
    pub total_memory_mb: usize,
    pub free_memory_mb: usize,
    pub cuda_cores: u32,
    pub driver_version: String,
}

impl GpuInfo {
    pub fn get_current() -> Result<Self> {
        // This is a placeholder. In a real application, you would use
        // a CUDA binding to get actual device properties.
        // For example, using `cuda_driver` crate or similar.
        warn!("Detailed GPU information retrieval is a placeholder. Actual CUDA API calls are needed here.");
        Ok(GpuInfo::default())
    }
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
/// Selects the appropriate backend based on environment configuration and GPU capabilities.
///
/// This function replicates the exact logic from RustTA::select_backend()
/// with proper fallback handling and intelligent GPU selection:
/// - "cpu" -> CPU backend
/// - "gpu" -> GPU backend (falls back to CPU if GPU unavailable or not suitable)
/// - "adaptive" -> Adaptive backend (falls back to CPU if unavailable)
/// - default -> Adaptive backend (falls back to CPU if unavailable)
pub fn select_backend() -> BackendSelectionResult {
    let config = GpuConfig::get();
    let requested_device = get_requested_device();

    match requested_device.as_deref() {
        Some("cpu") => {
            info!("Explicitly requested CPU backend.");
            (Box::new(CpuBackend::new()), "cpu")
        }
        Some("gpu") => {
            info!("Explicitly requested GPU backend.");
            if is_gpu_available() {
                match try_create_gpu_backend() {
                    Ok(backend) => {
                        info!("Successfully created GPU backend.");
                        (Box::new(backend), "gpu")
                    }
                    Err(e) => {
                        error!("Failed to create GPU backend, falling back to CPU: {}", e);
                        (Box::new(CpuBackend::new()), "cpu")
                    }
                }
            } else {
                warn!("GPU requested but not available or suitable, falling back to CPU.");
                (Box::new(CpuBackend::new()), "cpu")
            }
        }
        Some("adaptive") | _ => {
            // Default to adaptive if not specified or invalid
            info!("Requested Adaptive backend or defaulting to Adaptive.");
            match AdaptiveBackend::new() {
                Ok(backend) => {
                    info!("Successfully created Adaptive backend.");
                    (Box::new(backend), "adaptive")
                }
                Err(e) => {
                    error!(
                        "Failed to create Adaptive backend, falling back to CPU: {}",
                        e
                    );
                    (Box::new(CpuBackend::new()), "cpu")
                }
            }
        }
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

/// Selects the appropriate ML backend based on GPU availability.
///
/// This function attempts to create a GPU-accelerated ML backend if a GPU is available.
/// If no GPU is available or initialization fails, it falls back to the CPU-only ML backend.
///
/// # Returns
///
/// A tuple containing:
/// - `Box<dyn MLBackend>`: A boxed trait object for the selected ML backend.
/// - `&'static str`: A string indicating the type of backend selected ("gpu" or "cpu").
///
/// # GPU Detection
///
/// Uses `PartialGpuBackend::is_available()` to detect GPU availability.
///
/// # Example
///
/// ```rust
/// use rust_indicators::utils::backend_selection::select_ml_backend;
/// use rust_indicators::ml::traits::MLBackend;
///
/// let (ml_backend, backend_type) = select_ml_backend();
/// println!("Selected ML backend: {}", backend_type);
///
/// // You can now use ml_backend for ML operations
/// // assert!(ml_backend.is_trained()); // Example usage
/// ```
/// Selects the appropriate ML backend based on GPU availability and configuration.
///
/// This function attempts to create a GPU-accelerated ML backend if a GPU is available
/// and suitable according to the `GpuConfig`. If no GPU is available, initialization fails,
/// or the GPU is not suitable for the current workload (e.g., dataset size too small),
/// it falls back to the CPU-only ML backend.
///
/// # Arguments
///
/// * `dataset_size` - The size of the dataset to be processed, used for performance heuristics.
///
/// # Returns
///
/// A tuple containing:
/// - `Arc<dyn MLBackend>`: A boxed trait object for the selected ML backend.
/// - `&'static str`: A string indicating the type of backend selected ("gpu" or "cpu").
///
/// # GPU Detection
///
/// Uses `is_gpu_available()` to detect GPU availability and suitability.
pub fn select_ml_backend(dataset_size: Option<usize>) -> (Arc<dyn MLBackend>, &'static str) {
    let config = GpuConfig::get();

    if !config.enabled {
        info!("GPU is disabled by configuration, selecting CPU ML backend.");
        return (Arc::new(CpuMLBackend::new()), "cpu");
    }

    if let Some(min_size) = config.min_dataset_size {
        if let Some(ds_size) = dataset_size {
            if ds_size < min_size {
                info!("Dataset size ({}) is below the minimum threshold ({}) for GPU, selecting CPU ML backend.", ds_size, min_size);
                return (Arc::new(CpuMLBackend::new()), "cpu");
            }
        }
    }

    if is_gpu_available() {
        // Further performance heuristics could be added here, e.g.,
        // comparing estimated GPU performance vs CPU for a given workload.
        match try_create_gpu_backend() {
            Ok(backend) => {
                info!("Successfully created GPU ML backend.");
                (Arc::new(backend), "gpu")
            }
            Err(e) => {
                error!(
                    "Failed to create GPU ML backend, falling back to CPU: {}",
                    e
                );
                (Arc::new(CpuMLBackend::new()), "cpu")
            }
        }
    } else {
        warn!("GPU not available or suitable, selecting CPU ML backend.");
        (Arc::new(CpuMLBackend::new()), "cpu")
    }
}
