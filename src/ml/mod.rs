//! Machine learning utilities and models
//!
//! This module contains machine learning components including
//! model definitions, training utilities, prediction interfaces,
//! shared ML components, unified ML models, and core ML traits for backend implementations.

pub mod model;
pub mod traits;
pub mod components;
pub mod models;

pub use model::RustMLModel;
pub use traits::{MLBackend, LabelGenerator, CrossValidator, Predictor, UnifiedMLBackend};
pub use components::{
    PurgedCrossValidator, PatternAwareCrossValidator,
    VolatilityWeighting, PatternWeighting, SampleWeightCalculator,
    TripleBarrierLabeler, PatternLabeler, ComponentLabelGenerator,
    ConfidencePredictor, BatchPredictor, PredictionEngine,
};
pub use models::{
    PatternClassifier, TradingClassifier, UnifiedClassifier,
};

// Re-export ClassifierMode from unified_classifier for Python bindings
pub use models::unified_classifier::ClassifierMode;
