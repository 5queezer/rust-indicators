//! CPU-based implementations of technical indicators
//!
//! This module contains optimized CPU implementations
//! for technical indicator calculations.

pub mod backend;
pub mod implementations;
pub use backend::CpuBackend;
