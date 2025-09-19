//! CPU-based implementations of technical indicators
//!
//! This module contains optimized CPU implementations
//! for technical indicator calculations.

pub mod backend;
pub mod implementations;
pub mod ml_backend; // New module for CPU ML backend
pub use backend::CpuBackend;
pub use ml_backend::CpuMLBackend; // Export CpuMLBackend
