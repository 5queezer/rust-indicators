//! GPU-based implementations of technical indicators
//!
//! This module contains GPU-accelerated implementations
//! for technical indicator calculations using compute shaders.

pub mod backend;
#[cfg(feature = "gpu")]
pub mod implementations;
pub use backend::PartialGpuBackend;
