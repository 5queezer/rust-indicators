//! Core traits for machine learning backends and models
//!
//! This module defines the fundamental traits that all ML backends and models must implement.
//! The trait system enables polymorphic model selection and consistent API across
//! different ML strategies (pattern recognition, trading classification, etc.).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Type alias for the return type of predict_batch
pub type BatchPredictionOutput = (Py<PyArray1<i32>>, Py<PyArray1<f32>>);

/// Core trait for machine learning computation backends
///
/// This trait defines the interface that all ML backends must implement.
/// It provides a consistent API for training models, generating predictions,
/// and managing ML workflows while allowing different backends to optimize
/// computation strategies.
///
/// # Backend Implementations
///
/// - **PatternMLBackend**: Specialized for pattern recognition and ensemble methods
/// - **ClassificationMLBackend**: Optimized for trading signal classification
/// - **AdaptiveMLBackend**: Intelligent backend that selects optimal ML strategy
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` to support:
/// - Multi-threaded access from Python
/// - Safe sharing across thread boundaries
/// - Static lifetime for global backend instances
///
/// # Error Handling
///
/// All methods return `PyResult<T>` to properly propagate Python exceptions.
/// Implementations should handle computation errors gracefully and provide
/// meaningful error messages for debugging.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::ml::traits::MLBackend;
/// use numpy::{PyArray1, PyReadonlyArray2};
/// use pyo3::prelude::*;
/// use pyo3::exceptions::PyValueError;
/// use std::collections::HashMap;
///
/// struct MyMLBackend {
///     trained: bool,
///     feature_count: usize,
/// }
///
/// impl MLBackend for MyMLBackend {
///     fn train_model<'py>(&mut self, py: Python<'py>,
///         features: PyReadonlyArray2<'py, f32>,
///         labels: PyReadonlyArray1<'py, i32>)
///         -> PyResult<HashMap<String, f32>> {
///         let features_array = features.as_array();
///         let labels_array = labels.as_array();
///         
///         if features_array.nrows() != labels_array.len() {
///             return Err(PyValueError::new_err("Feature and label count mismatch"));
///         }
///         
///         self.trained = true;
///         self.feature_count = features_array.ncols();
///         
///         let mut metrics = HashMap::new();
///         metrics.insert("cv_mean".to_string(), 0.85);
///         metrics.insert("cv_std".to_string(), 0.05);
///         Ok(metrics)
///     }
///
///     fn predict_with_confidence<'py>(&self, py: Python<'py>,
///         features: PyReadonlyArray1<'py, f32>) -> PyResult<(i32, f32)> {
///         if !self.trained {
///             return Err(PyValueError::new_err("Model not trained"));
///         }
///         Ok((1, 0.75)) // Hold signal with 75% confidence
///     }
///
///     fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
///         if !self.trained {
///             return Err(PyValueError::new_err("Model not trained"));
///         }
///         let importance = vec![0.1; self.feature_count];
///         Ok(PyArray1::from_vec(py, importance).to_owned().into())
///     }
///
///     fn is_trained(&self) -> bool { self.trained }
///     fn reset_model(&mut self) { self.trained = false; }
/// }
/// ```
pub trait MLBackend: Send + Sync + 'static {
    /// Train a machine learning model
    ///
    /// Trains the ML model using the provided features and labels.
    /// Returns training metrics including cross-validation scores,
    /// feature importance, and model performance statistics.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 2D array of feature vectors (samples × features)
    /// - `labels`: 1D array of target labels
    ///
    /// # Returns
    /// `HashMap<String, f32>` containing training metrics:
    /// - "cv_mean": Mean cross-validation score
    /// - "cv_std": Standard deviation of CV scores
    /// - "n_features": Number of features used
    /// - "training_accuracy": Final training accuracy
    ///
    /// # Errors
    /// - `PyValueError`: If feature/label dimensions don't match
    /// - `PyRuntimeError`: If training fails due to computation errors
    fn train_model<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<HashMap<String, f32>>;

    /// Generate predictions with confidence scores
    ///
    /// Produces predictions for new feature vectors along with confidence scores.
    /// The confidence score indicates the model's certainty in its prediction.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 1D array of feature values for a single sample
    ///
    /// # Returns
    /// Tuple of (prediction_class, confidence_score) where:
    /// - prediction_class: Integer class prediction (0=sell, 1=hold, 2=buy)
    /// - confidence_score: Float confidence in [0, 1] range
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained or feature dimension mismatch
    /// - `PyRuntimeError`: If prediction computation fails
    fn predict_with_confidence<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)>;

    /// Get feature importance scores
    ///
    /// Returns the importance scores for each feature as determined during training.
    /// Higher scores indicate more important features for the model's decisions.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    ///
    /// # Returns
    /// `PyArray1<f32>` containing feature importance scores
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained
    fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>>;

    /// Check if the model has been trained
    ///
    /// # Returns
    /// `true` if the model has been successfully trained, `false` otherwise
    fn is_trained(&self) -> bool;

    /// Reset the model to untrained state
    ///
    /// Clears all trained parameters and resets the model to its initial state.
    /// Useful for retraining with new data or different parameters.
    fn reset_model(&mut self);
}

/// Trait for generating trading labels from price data
///
/// This trait defines methods for creating various types of trading labels
/// from OHLC price data. It supports different labeling strategies including
/// triple barrier method, pattern-based labels, and volatility-adjusted labels.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` for thread safety.
///
/// # Error Handling
///
/// All methods return `PyResult<T>` using proper PyO3 error handling patterns.
/// Common errors include dimension mismatches and invalid parameter values.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::ml::traits::LabelGenerator;
/// use numpy::{PyArray1, PyReadonlyArray1};
/// use pyo3::prelude::*;
/// use pyo3::exceptions::PyValueError;
///
/// struct MyLabelGenerator;
///
/// impl LabelGenerator for MyLabelGenerator {
///     fn create_triple_barrier_labels<'py>(&self, py: Python<'py>,
///         prices: PyReadonlyArray1<'py, f32>,
///         volatility: PyReadonlyArray1<'py, f32>,
///         profit_mult: f32, stop_mult: f32, max_hold: usize)
///         -> PyResult<Py<PyArray1<i32>>> {
///         let prices_slice = prices.as_slice()?;
///         let vol_slice = volatility.as_slice()?;
///         
///         if prices_slice.len() != vol_slice.len() {
///             return Err(PyValueError::new_err("Price and volatility arrays must have same length"));
///         }
///         
///         let labels = vec![1i32; prices_slice.len()]; // All hold signals
///         Ok(PyArray1::from_vec(py, labels).to_owned().into())
///     }
///
///     // ... implement other methods
/// #   fn create_pattern_labels<'py>(&self, py: Python<'py>, open_prices: PyReadonlyArray1<'py, f32>, high_prices: PyReadonlyArray1<'py, f32>, low_prices: PyReadonlyArray1<'py, f32>, close_prices: PyReadonlyArray1<'py, f32>, future_periods: usize, profit_threshold: f32, stop_threshold: f32) -> PyResult<Py<PyArray1<i32>>> { todo!() }
/// #   fn calculate_sample_weights<'py>(&self, py: Python<'py>, returns: PyReadonlyArray1<'py, f32>, volatility: Option<PyReadonlyArray1<'py, f32>>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// }
/// ```
pub trait LabelGenerator: Send + Sync + 'static {
    /// Create triple barrier labels
    ///
    /// Generates trading labels using the triple barrier method, which considers
    /// profit targets, stop losses, and maximum holding periods to determine
    /// optimal trading signals.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `prices`: Array of price values (typically closing prices)
    /// - `volatility`: Array of volatility estimates for each period
    /// - `profit_mult`: Profit target multiplier (e.g., 2.0 for 2x volatility)
    /// - `stop_mult`: Stop loss multiplier (e.g., 1.5 for 1.5x volatility)
    /// - `max_hold`: Maximum holding period in bars
    ///
    /// # Returns
    /// `PyArray1<i32>` containing labels:
    /// - 0: Sell signal (stop loss hit)
    /// - 1: Hold signal (no clear direction)
    /// - 2: Buy signal (profit target hit)
    ///
    /// # Formula
    /// For each bar i:
    /// - profit_target = price[i] * (1 + profit_mult * volatility[i])
    /// - stop_target = price[i] * (1 - stop_mult * volatility[i])
    /// - Check future prices up to max_hold periods
    ///
    /// # Errors
    /// - `PyValueError`: If array lengths don't match or invalid parameters
    fn create_triple_barrier_labels<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
        profit_mult: f32,
        stop_mult: f32,
        max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>>;

    /// Create pattern-based labels
    ///
    /// Generates labels based on pattern recognition signals combined with
    /// future price movements. Useful for training pattern recognition models.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `open_prices`: Array of opening prices
    /// - `high_prices`: Array of high prices
    /// - `low_prices`: Array of low prices
    /// - `close_prices`: Array of closing prices
    /// - `future_periods`: Number of periods to look ahead
    /// - `profit_threshold`: Minimum profit threshold for buy signal
    /// - `stop_threshold`: Maximum loss threshold for sell signal
    ///
    /// # Returns
    /// `PyArray1<i32>` containing pattern-based labels (0=sell, 1=hold, 2=buy)
    ///
    /// # Errors
    /// - `PyValueError`: If OHLC arrays have different lengths
    fn create_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        ohlc_data: OHLCData<'py>,
        params: PatternLabelingParams,
    ) -> PyResult<Py<PyArray1<i32>>>;

    /// Calculate sample weights
    ///
    /// Computes sample weights based on various factors such as volatility,
    /// return magnitude, and pattern rarity. Higher weights are assigned to
    /// more informative samples.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `returns`: Array of return values
    /// - `volatility`: Array of volatility estimates (optional)
    ///
    /// # Returns
    /// `PyArray1<f32>` containing sample weights
    ///
    /// # Formula
    /// Weight based on:
    /// - Return magnitude relative to recent average
    /// - Volatility normalization
    /// - Pattern rarity (if applicable)
    ///
    /// # Errors
    /// - `PyValueError`: If input arrays have invalid dimensions
    fn calculate_sample_weights<'py>(
        &self,
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
        volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>>;
}

/// Trait for cross-validation strategies
///
/// This trait defines methods for creating cross-validation splits that are
/// appropriate for time series data, including purged cross-validation and
/// embargo periods to prevent data leakage.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` for thread safety.
///
/// # Error Handling
///
/// Methods return `PyResult<T>` for proper error propagation to Python.
/// Common errors include insufficient sample sizes and invalid parameters.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::ml::traits::CrossValidator;
/// use pyo3::prelude::*;
/// use pyo3::exceptions::PyValueError;
///
/// struct MyValidator;
///
/// impl CrossValidator for MyValidator {
///     fn create_purged_cv_splits(&self, n_samples: usize, n_splits: usize, embargo_pct: f32)
///         -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
///         if n_samples < n_splits {
///             return Err(PyValueError::new_err("Not enough samples for splits"));
///         }
///         
///         let mut splits = Vec::new();
///         let fold_size = n_samples / n_splits;
///         
///         for i in 0..n_splits {
///             let test_start = i * fold_size;
///             let test_end = if i == n_splits - 1 { n_samples } else { (i + 1) * fold_size };
///             let test_indices: Vec<usize> = (test_start..test_end).collect();
///             let train_indices: Vec<usize> = (0..test_start).chain(test_end..n_samples).collect();
///             splits.push((train_indices, test_indices));
///         }
///         
///         Ok(splits)
///     }
///
///     // ... implement other methods
/// #   fn create_pattern_aware_cv_splits(&self, n_samples: usize, n_splits: usize, pattern_duration: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> { todo!() }
/// #   fn validate_cv_splits(&self, splits: &[(Vec<usize>, Vec<usize>)], min_train_size: usize, min_test_size: usize) -> bool { todo!() }
/// }
/// ```
pub trait CrossValidator: Send + Sync + 'static {
    /// Create purged cross-validation splits
    ///
    /// Generates cross-validation splits with embargo periods to prevent
    /// data leakage in time series data. Each test set is separated from
    /// training data by an embargo period.
    ///
    /// # Parameters
    /// - `n_samples`: Total number of samples
    /// - `n_splits`: Number of CV folds to create
    /// - `embargo_pct`: Embargo period as percentage of total samples
    ///
    /// # Returns
    /// Vector of (train_indices, test_indices) tuples for each fold
    ///
    /// # Algorithm
    /// 1. Divide data into n_splits consecutive folds
    /// 2. For each fold, use it as test set
    /// 3. Create embargo periods before and after test set
    /// 4. Use remaining data as training set
    ///
    /// # Errors
    /// - `PyValueError`: If n_samples < n_splits or invalid embargo_pct
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>>;

    /// Create pattern-aware CV splits
    ///
    /// Generates CV splits that account for pattern duration to prevent
    /// overlapping patterns between training and test sets.
    ///
    /// # Parameters
    /// - `n_samples`: Total number of samples
    /// - `n_splits`: Number of CV folds
    /// - `pattern_duration`: Duration of patterns in bars
    ///
    /// # Returns
    /// Vector of (train_indices, test_indices) tuples
    ///
    /// # Algorithm
    /// Similar to purged CV but uses pattern_duration * 2 as minimum embargo
    ///
    /// # Errors
    /// - `PyValueError`: If parameters are invalid
    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>>;

    /// Validate CV split quality
    ///
    /// Checks the quality of CV splits by ensuring adequate separation
    /// between training and test sets and balanced fold sizes.
    ///
    /// # Parameters
    /// - `splits`: Vector of CV splits to validate
    /// - `min_train_size`: Minimum required training set size
    /// - `min_test_size`: Minimum required test set size
    ///
    /// # Returns
    /// `true` if splits are valid, `false` otherwise
    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool;
}

/// Trait for making predictions and managing model inference
///
/// This trait defines methods for generating predictions, managing prediction
/// confidence, and providing model interpretability features.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` for thread safety.
///
/// # Error Handling
///
/// All methods use `PyResult<T>` for proper error handling and propagation
/// to Python. Common errors include untrained models and dimension mismatches.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::ml::traits::Predictor;
/// use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
/// use pyo3::prelude::*;
/// use pyo3::exceptions::PyValueError;
///
/// struct MyPredictor {
///     trained: bool,
///     confidence_threshold: f32,
/// }
///
/// impl Predictor for MyPredictor {
///     fn predict_single<'py>(&self, py: Python<'py>,
///         features: PyReadonlyArray1<'py, f32>)
///         -> PyResult<(i32, f32)> {
///         if !self.trained {
///             return Err(PyValueError::new_err("Model not trained"));
///         }
///         
///         let _features_slice = features.as_slice()?;
///         // Prediction logic here
///         Ok((1, 0.8)) // Hold signal with 80% confidence
///     }
///
///     fn predict_batch<'py>(&self, py: Python<'py>,
///         features: PyReadonlyArray2<'py, f32>)
///         -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
///         if !self.trained {
///             return Err(PyValueError::new_err("Model not trained"));
///         }
///         
///         let features_array = features.as_array();
///         let n_samples = features_array.nrows();
///         
///         let predictions = vec![1i32; n_samples]; // All hold signals
///         let confidences = vec![0.8f32; n_samples]; // 80% confidence
///         
///         Ok((
///             PyArray1::from_vec(py, predictions).to_owned().into(),
///             PyArray1::from_vec(py, confidences).to_owned().into()
///         ))
///     }
///
///     // ... implement other methods
/// #   fn predict_probabilities<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// #   fn get_prediction_explanation<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// #   fn set_confidence_threshold(&mut self, threshold: f32) -> PyResult<()> { Ok(()) }
/// #   fn get_confidence_threshold(&self) -> f32 { 0.5 }
/// }
/// ```
pub trait Predictor: Send + Sync + 'static {
    /// Make prediction for a single sample
    ///
    /// Generates a prediction and confidence score for a single feature vector.
    /// This is the core prediction method used for real-time inference.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 1D array of feature values
    ///
    /// # Returns
    /// Tuple of (prediction_class, confidence_score) where:
    /// - prediction_class: Integer class (0=sell, 1=hold, 2=buy)
    /// - confidence_score: Float confidence in [0, 1] range
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained or wrong feature dimensions
    /// - `PyRuntimeError`: If prediction computation fails
    fn predict_single<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)>;

    /// Make batch predictions
    ///
    /// Generates predictions for multiple samples efficiently.
    /// Useful for backtesting and batch processing scenarios.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 2D array of feature vectors (samples × features)
    ///
    /// # Returns
    /// Tuple of (predictions, confidences) as PyArray1 objects
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained or invalid dimensions
    /// - `PyRuntimeError`: If batch prediction fails
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<BatchPredictionOutput>;

    /// Get prediction probabilities
    ///
    /// Returns class probabilities for each possible outcome instead of
    /// just the most likely class. Useful for risk management and
    /// position sizing decisions.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 1D array of feature values
    ///
    /// # Returns
    /// `PyArray1<f32>` containing probabilities for each class [P(sell), P(hold), P(buy)]
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained or wrong dimensions
    fn predict_probabilities<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>>;

    /// Get model interpretability information
    ///
    /// Returns information about which features contributed most to a prediction.
    /// Useful for understanding model decisions and debugging.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `features`: 1D array of feature values
    ///
    /// # Returns
    /// `PyArray1<f32>` containing feature contributions to the prediction
    ///
    /// # Errors
    /// - `PyValueError`: If model not trained or interpretability not supported
    fn get_prediction_explanation<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>>;

    /// Set prediction confidence threshold
    ///
    /// Sets the minimum confidence threshold for making predictions.
    /// Predictions below this threshold will return "hold" signal.
    ///
    /// # Parameters
    /// - `threshold`: Confidence threshold in [0, 1] range
    ///
    /// # Errors
    /// - `PyValueError`: If threshold is outside valid range
    fn set_confidence_threshold(&mut self, threshold: f32) -> PyResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PyValueError::new_err(
                "Confidence threshold must be between 0.0 and 1.0",
            ));
        }
        self.set_confidence_threshold_unchecked(threshold);
        Ok(())
    }

    /// Set confidence threshold without validation (internal use)
    ///
    /// # Parameters
    /// - `threshold`: Confidence threshold value
    fn set_confidence_threshold_unchecked(&mut self, threshold: f32);

    /// Get current confidence threshold
    ///
    /// # Returns
    /// Current confidence threshold value
    fn get_confidence_threshold(&self) -> f32;
}

/// Unified ML trait combining all ML functionality
///
/// This trait combines all ML-related traits into a single interface for
/// implementations that provide complete ML functionality. It's useful for
/// creating comprehensive ML backends that handle the entire workflow.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` for thread safety.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::ml::traits::{MLBackend, LabelGenerator, CrossValidator, Predictor, UnifiedMLBackend};
/// use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
/// use pyo3::prelude::*;
/// use std::collections::HashMap;
///
/// struct MyUnifiedBackend {
///     trained: bool,
///     confidence_threshold: f32,
/// }
///
/// impl MLBackend for MyUnifiedBackend {
///     // Implement MLBackend methods...
/// #   fn train_model<'py>(&mut self, py: Python<'py>, features: PyReadonlyArray2<'py, f32>, labels: PyReadonlyArray1<'py, i32>) -> PyResult<HashMap<String, f32>> { todo!() }
/// #   fn predict_with_confidence<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<(i32, f32)> { todo!() }
/// #   fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// #   fn is_trained(&self) -> bool { todo!() }
/// #   fn reset_model(&mut self) { todo!() }
/// }
///
/// impl LabelGenerator for MyUnifiedBackend {
///     // Implement LabelGenerator methods...
/// #   fn create_triple_barrier_labels<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f32>, volatility: PyReadonlyArray1<'py, f32>, profit_mult: f32, stop_mult: f32, max_hold: usize) -> PyResult<Py<PyArray1<i32>>> { todo!() }
/// #   fn create_pattern_labels<'py>(&self, py: Python<'py>, open_prices: PyReadonlyArray1<'py, f32>, high_prices: PyReadonlyArray1<'py, f32>, low_prices: PyReadonlyArray1<'py, f32>, close_prices: PyReadonlyArray1<'py, f32>, future_periods: usize, profit_threshold: f32, stop_threshold: f32) -> PyResult<Py<PyArray1<i32>>> { todo!() }
/// #   fn calculate_sample_weights<'py>(&self, py: Python<'py>, returns: PyReadonlyArray1<'py, f32>, volatility: Option<PyReadonlyArray1<'py, f32>>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// }
///
/// impl CrossValidator for MyUnifiedBackend {
///     // Implement CrossValidator methods...
/// #   fn create_purged_cv_splits(&self, n_samples: usize, n_splits: usize, embargo_pct: f32) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> { todo!() }
/// #   fn create_pattern_aware_cv_splits(&self, n_samples: usize, n_splits: usize, pattern_duration: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> { todo!() }
/// #   fn validate_cv_splits(&self, splits: &[(Vec<usize>, Vec<usize>)], min_train_size: usize, min_test_size: usize) -> bool { todo!() }
/// }
///
/// impl Predictor for MyUnifiedBackend {
///     // Implement Predictor methods...
/// #   fn predict_single<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<(i32, f32)> { todo!() }
/// #   fn predict_batch<'py>(&self, py: Python<'py>, features: PyReadonlyArray2<'py, f32>) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> { todo!() }
/// #   fn predict_probabilities<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// #   fn get_prediction_explanation<'py>(&self, py: Python<'py>, features: PyReadonlyArray1<'py, f32>) -> PyResult<Py<PyArray1<f32>>> { todo!() }
/// #   fn set_confidence_threshold_unchecked(&mut self, threshold: f32) { self.confidence_threshold = threshold; }
/// #   fn get_confidence_threshold(&self) -> f32 { self.confidence_threshold }
/// }
///
/// impl UnifiedMLBackend for MyUnifiedBackend {}
/// ```
pub trait UnifiedMLBackend: MLBackend + LabelGenerator + CrossValidator + Predictor {}

// Blanket implementation for any type that implements all four traits
impl<T> UnifiedMLBackend for T where T: MLBackend + LabelGenerator + CrossValidator + Predictor {}

/// Struct to encapsulate OHLC price data for pattern labeling.
#[derive(Debug, Clone)]
pub struct OHLCData<'py> {
    pub open_prices: PyReadonlyArray1<'py, f32>,
    pub high_prices: PyReadonlyArray1<'py, f32>,
    pub low_prices: PyReadonlyArray1<'py, f32>,
    pub close_prices: PyReadonlyArray1<'py, f32>,
}

/// Struct to encapsulate parameters for pattern-based label generation.
#[derive(Debug, Clone, Copy)]
pub struct PatternLabelingParams {
    pub future_periods: usize,
    pub profit_threshold: f32,
    pub stop_threshold: f32,
}
