//! Financial time series data structures and operations
//! 
//! This module provides the core financial data structures for time series analysis,
//! implementing efficient operations for financial data processing based on López de Prado's
//! "Advances in Financial Machine Learning" methodologies.
//! 
//! # Key Features
//! 
//! - Zero-copy time series operations where possible
//! - Memory-efficient rolling window iterators
//! - Missing data handling strategies
//! - Index alignment for multi-series operations
//! - Business day calendar support
//! 
//! # Example
//! 
//! ```rust
//! use rust_indicators::financial::FinancialSeries;
//! use time::OffsetDateTime;
//! use ndarray::Array1;
//! 
//! // Create a financial time series
//! let timestamps = vec![
//!     OffsetDateTime::now_utc(),
//!     OffsetDateTime::now_utc() + time::Duration::days(1),
//! ];
//! let values = Array1::from(vec![100.0, 102.0]);
//! let series = FinancialSeries::new(timestamps, values);
//! 
//! // Calculate percentage changes
//! let pct_changes = series.pct_change(1);
//! ```

pub mod series;
pub mod bars;

pub use series::FinancialSeries;
pub use bars::{BarType, Bar, Tick, BarBuilder, ImbalanceTracker};