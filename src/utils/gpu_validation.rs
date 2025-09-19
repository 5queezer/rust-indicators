//! GPU Detection and Validation Utilities
//!
//! This module provides functions for detecting and validating GPU capabilities
//! at runtime, ensuring that the system can gracefully handle different hardware
//! and driver configurations.

use crate::config::gpu_config::GpuConfig;
use log::{info, warn};

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub driver_version: String,
    pub memory_mb: usize,
    pub cuda_cores: Option<u32>,
}

/// Detects available GPUs and returns information about them.
/// NOTE: This is a placeholder and needs a real implementation.
pub fn detect_gpus() -> Vec<GpuInfo> {
    // Placeholder: In a real implementation, you would query the system.
    vec![GpuInfo {
        name: "GeForce RTX 4090".to_string(),
        driver_version: "525.60.11".to_string(),
        memory_mb: 24576,
        cuda_cores: Some(16384),
    }]
}

/// Validates the GPU configuration against the detected hardware.
pub fn validate_gpu_config(config: &GpuConfig, gpus: &[GpuInfo]) -> bool {
    if !config.enabled {
        info!("GPU support is disabled in the configuration.");
        return false;
    }

    if gpus.is_empty() {
        warn!("GPU support is enabled, but no compatible GPUs were found.");
        return false;
    }

    let primary_gpu = &gpus[0];
    info!(
        "Found GPU: {} with {}MB of memory.",
        primary_gpu.name, primary_gpu.memory_mb
    );

    if let Some(memory_limit) = config.memory_limit_mb {
        if primary_gpu.memory_mb < memory_limit {
            warn!(
                "GPU memory ({}MB) is less than the configured limit ({}MB).",
                primary_gpu.memory_mb, memory_limit
            );
            return false;
        }
    }

    true
}
