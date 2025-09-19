//! Unified ML models integrating pattern recognition and trading classification
//!
//! This module provides unified ML models that integrate both pattern recognition and
//! trading classification using the shared components from src/ml/components/. The models
//! eliminate code duplication while maintaining all functionality from the original examples.
//!
//! # Architecture
//!
//! The models are built using a layered architecture:
//! - **Shared Components**: Reusable ML components (label generation, prediction, etc.)
//! - **Specialized Models**: Pattern and trading classifiers with specific functionality
//! - **Unified Model**: Combined classifier that can switch between modes
//!
//! # Models
//!
//! - [`PatternClassifier`]: Pattern recognition classifier using ensemble methods
//! - [`TradingClassifier`]: Scientific trading classifier with purged cross-validation
//! - [`UnifiedClassifier`]: Combined classifier supporting both pattern and trading modes
//!
//! # Thread Safety
//!
//! All models implement `Send + Sync` bounds for thread-safe access from Python.
//!
//! # Device Management
//!
//! Models support both CPU and CUDA execution following rust_indicators patterns:
//! - Automatic device selection (CUDA if available, CPU fallback)
//! - Proper device management with error handling
//! - Consistent device usage across all operations

pub mod pattern_classifier;
pub mod trading_classifier;
pub mod unified_classifier;

pub use pattern_classifier::PatternClassifier;
pub use trading_classifier::TradingClassifier;
pub use unified_classifier::UnifiedClassifier;

// Re-export shared components for convenience
pub use crate::ml::components::{
    BatchPredictor, ComponentLabelGenerator, ConfidencePredictor, PatternAwareCrossValidator,
    PatternLabeler, PatternWeighting, PredictionEngine, PurgedCrossValidator,
    SampleWeightCalculator, TripleBarrierLabeler, VolatilityWeighting,
};

// Re-export traits
pub use crate::ml::traits::{
    CrossValidator, LabelGenerator, MLBackend, Predictor, UnifiedMLBackend,
};
