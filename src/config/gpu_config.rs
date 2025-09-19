//! GPU Configuration Module
//!
//! This module defines the `GpuConfig` struct and provides functionality
//! for loading GPU-related settings from environment variables and
//! configuration files.

use anyhow::Result;
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

// Define a static configuration that can be loaded once
static GPU_CONFIG: Lazy<GpuConfig> = Lazy::new(GpuConfig::load);

/// Configuration for GPU operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub memory_limit_mb: Option<usize>,
    pub min_dataset_size: Option<usize>,
    pub performance_threshold_ms: Option<u64>,
    pub driver_version_min: Option<String>,
    pub cuda_cores_min: Option<u32>,
    pub pub_config_path: Option<String>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Default to enabled if not specified
            memory_limit_mb: None,
            min_dataset_size: None,
            performance_threshold_ms: None,
            driver_version_min: None,
            cuda_cores_min: None,
            pub_config_path: None,
        }
    }
}

impl GpuConfig {
    /// Loads GPU configuration from environment variables or a configuration file.
    pub fn load() -> Self {
        let mut config = GpuConfig::default();

        // 1. Load from environment variables
        if let Ok(val) = env::var("ML_GPU_ENABLED") {
            config.enabled = val.to_lowercase() == "true" || val == "1";
        }
        if let Ok(val) = env::var("ML_GPU_MEMORY_LIMIT_MB") {
            if let Ok(mem) = val.parse::<usize>() {
                config.memory_limit_mb = Some(mem);
            } else {
                warn!("Invalid value for ML_GPU_MEMORY_LIMIT_MB: {}", val);
            }
        }
        if let Ok(val) = env::var("ML_GPU_MIN_DATASET_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.min_dataset_size = Some(size);
            } else {
                warn!("Invalid value for ML_GPU_MIN_DATASET_SIZE: {}", val);
            }
        }
        if let Ok(val) = env::var("ML_GPU_PERFORMANCE_THRESHOLD_MS") {
            if let Ok(threshold) = val.parse::<u64>() {
                config.performance_threshold_ms = Some(threshold);
            } else {
                warn!("Invalid value for ML_GPU_PERFORMANCE_THRESHOLD_MS: {}", val);
            }
        }
        if let Ok(val) = env::var("ML_GPU_DRIVER_VERSION_MIN") {
            config.driver_version_min = Some(val);
        }
        if let Ok(val) = env::var("ML_GPU_CUDA_CORES_MIN") {
            if let Ok(cores) = val.parse::<u32>() {
                config.cuda_cores_min = Some(cores);
            } else {
                warn!("Invalid value for ML_GPU_CUDA_CORES_MIN: {}", val);
            }
        }
        if let Ok(val) = env::var("ML_GPU_CONFIG_PATH") {
            config.pub_config_path = Some(val);
        }

        // 2. Load from configuration file if specified
        if let Some(config_path_str) = &config.pub_config_path {
            let config_path = Path::new(config_path_str);
            if config_path.exists() {
                match fs::read_to_string(config_path) {
                    Ok(content) => {
                        match toml::from_str::<GpuConfig>(&content) {
                            Ok(file_config) => {
                                // Overlay file config over env vars (env vars take precedence)
                                if file_config.enabled != GpuConfig::default().enabled {
                                    config.enabled = file_config.enabled;
                                }
                                if file_config.memory_limit_mb.is_some() {
                                    config.memory_limit_mb = file_config.memory_limit_mb;
                                }
                                if file_config.min_dataset_size.is_some() {
                                    config.min_dataset_size = file_config.min_dataset_size;
                                }
                                if file_config.performance_threshold_ms.is_some() {
                                    config.performance_threshold_ms =
                                        file_config.performance_threshold_ms;
                                }
                                if file_config.driver_version_min.is_some() {
                                    config.driver_version_min = file_config.driver_version_min;
                                }
                                if file_config.cuda_cores_min.is_some() {
                                    config.cuda_cores_min = file_config.cuda_cores_min;
                                }
                                info!("Loaded GPU configuration from file: {}", config_path_str);
                            }
                            Err(e) => error!(
                                "Failed to parse GPU configuration file {}: {}",
                                config_path_str, e
                            ),
                        }
                    }
                    Err(e) => error!(
                        "Failed to read GPU configuration file {}: {}",
                        config_path_str, e
                    ),
                }
            } else {
                warn!("GPU configuration file not found: {}", config_path_str);
            }
        }

        info!("Final GPU Configuration: {:?}", config);
        config
    }

    /// Returns a reference to the globally loaded GPU configuration.
    pub fn get() -> &'static Self {
        &GPU_CONFIG
    }
}
