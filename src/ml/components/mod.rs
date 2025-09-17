//! Shared ML components for eliminating code duplication
//!
//! This module provides reusable ML components that can be shared across different
//! model implementations, following the patterns established in the rust_indicators
//! codebase with proper safety using extract_safe! macro and comprehensive error handling.

pub mod cross_validation;
pub mod sample_weighting;
pub mod label_generation;
pub mod prediction;

pub use cross_validation::{PurgedCrossValidator, PatternAwareCrossValidator};
pub use sample_weighting::{VolatilityWeighting, PatternWeighting, SampleWeightCalculator};
pub use label_generation::{TripleBarrierLabeler, PatternLabeler, ComponentLabelGenerator};
pub use prediction::{ConfidencePredictor, BatchPredictor, PredictionEngine};