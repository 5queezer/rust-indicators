//! Core traits and types for the indicators library
//!
//! This module contains the fundamental traits and data structures
//! that define the interface for technical indicators.

pub mod bar;
pub use bar::*;

pub mod financial_series;
pub use financial_series::*;

pub mod traits;
pub use traits::*;
