//! Testing and benchmark utilities
//!
//! This module contains utilities for testing, benchmarking,
//! backend selection, and other development support functions.

pub mod backend_selection;
pub mod benchmarking;
pub mod gpu_validation;
pub mod macros;
pub mod performance;
pub mod testing;

// Re-export the delegation macro for easy access
// Note: #[macro_export] exports macros at crate root, not module level
pub use crate::delegate_indicator;
