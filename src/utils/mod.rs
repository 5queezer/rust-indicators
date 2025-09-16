//! Testing and benchmark utilities
//!
//! This module contains utilities for testing, benchmarking,
//! backend selection, and other development support functions.

pub mod testing;
pub mod backend_selection;
pub mod macros;
pub mod benchmarking;

// Re-export the delegation macro for easy access
// Note: #[macro_export] exports macros at crate root, not module level
pub use crate::delegate_indicator;
