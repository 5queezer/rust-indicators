// File: rust_indicators/src/lib.rs
use pyo3::prelude::*;

// Conditionally use mimalloc when the feature is enabled
#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod backends;
pub mod config;
/// Rust-powered technical analysis indicators for FreqTrade
/// Provides 10-100x performance improvement over Python equivalents
pub mod core;
pub mod features;
pub mod financial;
pub mod indicators;
pub mod ml;
pub mod utils;

// Re-export the main structs for Python bindings
pub use features::RustFeatures;
pub use financial::FinancialSeries;
pub use indicators::RustTA;
pub use ml::{ClassifierMode, PatternClassifier, TradingClassifier, UnifiedClassifier};

/// Python module definition
#[pymodule]
mod rust_indicators {
    #[pymodule_export]
    use super::RustTA;

    #[pymodule_export]
    use super::RustFeatures;

    #[pymodule_export]
    use super::PatternClassifier;

    #[pymodule_export]
    use super::TradingClassifier;

    #[pymodule_export]
    use super::UnifiedClassifier;

    #[pymodule_export]
    use super::ClassifierMode;
}
