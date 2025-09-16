// File: rust_indicators/src/lib.rs
use pyo3::prelude::*;

// Conditionally use mimalloc when the feature is enabled
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Rust-powered technical analysis indicators for FreqTrade
/// Provides 10-100x performance improvement over Python equivalents
pub mod backend;
pub mod backend_cpu;
pub mod cpu_impls;
pub mod indicators;
pub mod features;
pub mod ml_model;


// Re-export the main structs for Python bindings
pub use indicators::RustTA;
pub use features::RustFeatures;
pub use ml_model::RustMLModel;

/// Python module definition
#[pymodule]
mod rust_indicators {
    #[pymodule_export]
    use super::RustTA;
    
    #[pymodule_export]
    use super::RustFeatures;
    
    #[pymodule_export]
    use super::RustMLModel;
}
