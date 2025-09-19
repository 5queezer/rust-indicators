//! # Error Handling and Result Types
//!
//! Unified error handling and result types for ML components to provide
//! consistent error reporting and reduce code duplication across models.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::fmt;

/// Unified ML error types for consistent error handling across all models
#[derive(Debug, Clone)]
pub enum MLError {
    /// Input validation errors
    ValidationError(String),
    /// Training-related errors
    TrainingError(String),
    /// Prediction-related errors
    PredictionError(String),
    /// Advanced Cross-Validation validation errors
    Phase4Error(String),
    /// Cross-validation errors
    CrossValidationError(String),
    /// Configuration errors
    ConfigurationError(String),
    /// Data processing errors
    DataError(String),
    /// Model state errors
    ModelStateError(String),
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
            MLError::TrainingError(msg) => write!(f, "Training Error: {}", msg),
            MLError::PredictionError(msg) => write!(f, "Prediction Error: {}", msg),
            MLError::Phase4Error(msg) => write!(f, "Advanced Cross-Validation Error: {}", msg),
            MLError::CrossValidationError(msg) => write!(f, "Cross-Validation Error: {}", msg),
            MLError::ConfigurationError(msg) => write!(f, "Configuration Error: {}", msg),
            MLError::DataError(msg) => write!(f, "Data Error: {}", msg),
            MLError::ModelStateError(msg) => write!(f, "Model State Error: {}", msg),
        }
    }
}

impl std::error::Error for MLError {}

impl From<MLError> for PyErr {
    fn from(err: MLError) -> PyErr {
        match err {
            MLError::ValidationError(msg) => PyValueError::new_err(msg),
            MLError::TrainingError(msg) => PyRuntimeError::new_err(msg),
            MLError::PredictionError(msg) => PyRuntimeError::new_err(msg),
            MLError::Phase4Error(msg) => PyRuntimeError::new_err(msg),
            MLError::CrossValidationError(msg) => PyValueError::new_err(msg),
            MLError::ConfigurationError(msg) => PyValueError::new_err(msg),
            MLError::DataError(msg) => PyTypeError::new_err(msg),
            MLError::ModelStateError(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Result type alias for ML operations
pub type MLResult<T> = Result<T, MLError>;

/// Standardized training results across all models
#[derive(Debug, Clone)]
pub struct TrainingResults {
    pub cv_mean: f32,
    pub cv_std: f32,
    pub n_splits: usize,
    pub training_time_ms: u64,
    pub model_type: String,
    pub additional_metrics: std::collections::HashMap<String, f32>,
}

impl TrainingResults {
    /// Create new training results
    pub fn new(
        cv_mean: f32,
        cv_std: f32,
        n_splits: usize,
        training_time_ms: u64,
        model_type: String,
    ) -> Self {
        Self {
            cv_mean,
            cv_std,
            n_splits,
            training_time_ms,
            model_type,
            additional_metrics: std::collections::HashMap::new(),
        }
    }

    /// Add additional metric
    pub fn with_metric(mut self, key: String, value: f32) -> Self {
        self.additional_metrics.insert(key, value);
        self
    }

    /// Convert to HashMap for Python compatibility
    pub fn to_dict(&self) -> std::collections::HashMap<String, f32> {
        let mut dict = std::collections::HashMap::new();
        dict.insert("cv_mean".to_string(), self.cv_mean);
        dict.insert("cv_std".to_string(), self.cv_std);
        dict.insert("n_splits".to_string(), self.n_splits as f32);
        dict.insert("training_time_ms".to_string(), self.training_time_ms as f32);

        // Add additional metrics
        dict.extend(self.additional_metrics.clone());

        dict
    }
}

/// Standardized prediction results
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub prediction: i32,
    pub confidence: f32,
    pub model_type: String,
    pub processing_time_us: u64,
}

impl PredictionResult {
    pub fn new(
        prediction: i32,
        confidence: f32,
        model_type: String,
        processing_time_us: u64,
    ) -> Self {
        Self {
            prediction,
            confidence,
            model_type,
            processing_time_us,
        }
    }
}

/// Error handling utilities
pub struct ErrorHandler;

impl ErrorHandler {
    /// Validate input dimensions
    pub fn validate_dimensions(
        features_shape: (usize, usize),
        labels_len: usize,
        context: &str,
    ) -> MLResult<()> {
        let (n_samples, _) = features_shape;
        if n_samples != labels_len {
            return Err(MLError::ValidationError(format!(
                "{}: Feature samples ({}) != label samples ({})",
                context, n_samples, labels_len
            )));
        }
        Ok(())
    }

    /// Validate model is trained
    pub fn validate_trained(is_trained: bool, model_type: &str) -> MLResult<()> {
        if !is_trained {
            return Err(MLError::ModelStateError(format!(
                "{} model not trained. Call train method first.",
                model_type
            )));
        }
        Ok(())
    }

    /// Validate sample count
    pub fn validate_sample_count(
        n_samples: usize,
        min_samples: usize,
        context: &str,
    ) -> MLResult<()> {
        if n_samples < min_samples {
            return Err(MLError::ValidationError(format!(
                "{}: Insufficient samples ({} < {})",
                context, n_samples, min_samples
            )));
        }
        Ok(())
    }

    /// Validate feature count
    pub fn validate_feature_count(actual: usize, expected: usize, context: &str) -> MLResult<()> {
        if actual != expected {
            return Err(MLError::ValidationError(format!(
                "{}: Feature count mismatch (expected {}, got {})",
                context, expected, actual
            )));
        }
        Ok(())
    }

    /// Validate confidence threshold
    pub fn validate_confidence_threshold(threshold: f32) -> MLResult<f32> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(MLError::ConfigurationError(format!(
                "Confidence threshold must be in [0.0, 1.0], got {}",
                threshold
            )));
        }
        Ok(threshold)
    }

    /// Validate Advanced Cross-Validation configuration
    pub fn validate_advanced_cross_validation_config(
        n_samples: usize,
        n_groups: usize,
        test_groups: usize,
        min_train_size: usize,
        min_test_size: usize,
    ) -> MLResult<()> {
        if test_groups >= n_groups {
            return Err(MLError::ConfigurationError(format!(
                "test_groups ({}) must be < n_groups ({})",
                test_groups, n_groups
            )));
        }

        let min_required = min_train_size + min_test_size;
        if n_samples < min_required {
            return Err(MLError::ValidationError(format!(
                "Insufficient samples for Advanced Cross-Validation: {} < {} (min_train + min_test)",
                n_samples, min_required
            )));
        }

        Ok(())
    }

    /// Convert PyErr to MLError
    pub fn from_py_err(err: PyErr, context: &str) -> MLError {
        MLError::DataError(format!("{}: {}", context, err))
    }

    /// Create training error with context
    pub fn training_error(msg: &str, context: &str) -> MLError {
        MLError::TrainingError(format!("{}: {}", context, msg))
    }

    /// Create prediction error with context
    pub fn prediction_error(msg: &str, context: &str) -> MLError {
        MLError::PredictionError(format!("{}: {}", context, msg))
    }

    /// Create Advanced Cross-Validation error with context
    pub fn advanced_cross_validation_error(msg: &str, context: &str) -> MLError {
        MLError::Phase4Error(format!("{}: {}", context, msg))
    }
}

/// Performance tracking utilities
pub struct PerformanceTracker {
    start_time: std::time::Instant,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub fn elapsed_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result builders for consistent result creation
pub struct ResultBuilder;

impl ResultBuilder {
    /// Build training results with performance tracking
    pub fn training_results(
        cv_scores: &[f32],
        model_type: &str,
        tracker: &PerformanceTracker,
    ) -> TrainingResults {
        let cv_mean = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let cv_std = {
            let variance = cv_scores
                .iter()
                .map(|&x| (x - cv_mean).powi(2))
                .sum::<f32>()
                / cv_scores.len() as f32;
            variance.sqrt()
        };

        TrainingResults::new(
            cv_mean,
            cv_std,
            cv_scores.len(),
            tracker.elapsed_ms(),
            model_type.to_string(),
        )
    }

    /// Build prediction result with performance tracking
    pub fn prediction_result(
        prediction: i32,
        confidence: f32,
        model_type: &str,
        tracker: &PerformanceTracker,
    ) -> PredictionResult {
        PredictionResult::new(
            prediction,
            confidence,
            model_type.to_string(),
            tracker.elapsed_us(),
        )
    }

    /// Build error result with context
    pub fn error_result<T>(error: MLError) -> MLResult<T> {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_error_display() {
        let error = MLError::ValidationError("test message".to_string());
        assert_eq!(error.to_string(), "Validation Error: test message");
    }

    #[test]
    fn test_training_results_creation() {
        let results = TrainingResults::new(0.85, 0.05, 5, 1000, "TestModel".to_string());
        assert_eq!(results.cv_mean, 0.85);
        assert_eq!(results.cv_std, 0.05);
        assert_eq!(results.n_splits, 5);
        assert_eq!(results.training_time_ms, 1000);
        assert_eq!(results.model_type, "TestModel");
    }

    #[test]
    fn test_error_handler_validation() {
        assert!(ErrorHandler::validate_dimensions((100, 10), 100, "test").is_ok());
        assert!(ErrorHandler::validate_dimensions((100, 10), 50, "test").is_err());

        assert!(ErrorHandler::validate_trained(true, "TestModel").is_ok());
        assert!(ErrorHandler::validate_trained(false, "TestModel").is_err());

        assert!(ErrorHandler::validate_confidence_threshold(0.5).is_ok());
        assert!(ErrorHandler::validate_confidence_threshold(1.5).is_err());
    }

    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(tracker.elapsed_ms() >= 1);
        assert!(tracker.elapsed_us() >= 1000);
    }

    #[test]
    fn test_result_builder() {
        let scores = vec![0.8, 0.85, 0.75, 0.9];
        let tracker = PerformanceTracker::new();
        let results = ResultBuilder::training_results(&scores, "TestModel", &tracker);

        assert_eq!(results.model_type, "TestModel");
        assert_eq!(results.n_splits, 4);
        assert!(results.cv_mean > 0.0);
        assert!(results.cv_std >= 0.0);
    }
}
