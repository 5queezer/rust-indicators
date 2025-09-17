//! Pattern recognition classifier using shared components
//!
//! This module provides a pattern recognition classifier that integrates functionality
//! from pattern_model_example.rs while using the shared components to eliminate code
//! duplication. It implements ensemble methods for pattern-based trading signals.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray};
use std::collections::HashMap;

use crate::extract_safe;
use crate::ml::traits::{MLBackend, LabelGenerator, CrossValidator, Predictor};
use crate::ml::components::{
    PatternLabeler, PatternWeighting, PatternAwareCrossValidator, PredictionEngine,
};

/// Pattern recognition classifier with ensemble methods
///
/// This classifier specializes in pattern recognition using ensemble methods and
/// pattern-specific weighting strategies. It integrates all functionality from
/// PatternRecognitionClassifier while using shared components.
#[pyclass]
pub struct PatternClassifier {
    // Pattern-specific components
    pattern_labeler: PatternLabeler,
    pattern_weighting: PatternWeighting,
    cross_validator: PatternAwareCrossValidator,
    prediction_engine: PredictionEngine,
    
    // Pattern ensemble data
    pattern_weights: Option<HashMap<String, f32>>,
    pattern_importance: HashMap<String, f32>,
    
    // Training state
    trained: bool,
    pattern_names: Vec<String>,
    confidence_threshold: f32,
    
    // Cross-validation splits
    cv_splits: Vec<(Vec<usize>, Vec<usize>)>,
    sample_weights: Vec<f32>,
}

#[pymethods]
impl PatternClassifier {
    /// Create a new pattern classifier
    #[new]
    fn new(pattern_names: Vec<String>) -> PyResult<Self> {
        Ok(PatternClassifier {
            pattern_labeler: PatternLabeler::default(),
            pattern_weighting: PatternWeighting::default(),
            cross_validator: PatternAwareCrossValidator::default(),
            prediction_engine: PredictionEngine::default(),
            pattern_weights: None,
            pattern_importance: HashMap::new(),
            trained: false,
            pattern_names,
            confidence_threshold: 0.6,
            cv_splits: Vec::new(),
            sample_weights: Vec::new(),
        })
    }

    /// Train the pattern ensemble model
    fn train_pattern_ensemble(
        &mut self,
        pattern_features: PyReadonlyArray2<f32>,
        price_features: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        pattern_names: Vec<String>,
    ) -> PyResult<HashMap<String, f32>> {
        let pattern_array = pattern_features.as_array();
        let _price_array = price_features.as_array();
        let y_array = y.as_array();
        let (n_samples, n_patterns) = pattern_array.dim();

        if y_array.len() != n_samples {
            return Err(PyValueError::new_err("Feature and label count mismatch"));
        }

        // Initialize sample weights if not set
        if self.sample_weights.len() != n_samples {
            self.sample_weights = vec![1.0; n_samples];
        }

        // Create pattern-aware cross-validation splits
        self.cv_splits = self.cross_validator.create_default_pattern_splits(n_samples, Some(3))?;
        self.pattern_names = pattern_names;

        let mut cv_scores = Vec::new();
        let mut pattern_performances = HashMap::new();

        // Initialize pattern importance
        for pattern_name in &self.pattern_names {
            self.pattern_importance.insert(pattern_name.clone(), 0.0);
        }

        // Cross-validation training
        for (_train_idx, test_idx) in &self.cv_splits {
            let fold_score = self.evaluate_fold(&pattern_array, &y_array, test_idx)?;
            cv_scores.push(fold_score);

            // Calculate individual pattern performance
            for (i, pattern_name) in self.pattern_names.iter().enumerate() {
                if i < n_patterns {
                    let pattern_score = fold_score; // Simplified
                    let current_score = pattern_performances.get(pattern_name).unwrap_or(&0.0);
                    pattern_performances.insert(pattern_name.clone(), current_score + pattern_score);
                }
            }
        }

        // Average pattern performances and update importance
        for (pattern_name, total_score) in pattern_performances {
            let avg_score = total_score / self.cv_splits.len() as f32;
            self.pattern_importance.insert(pattern_name, avg_score);
        }

        // Create ensemble weights
        self.pattern_weights = Some(self.calculate_pattern_weights()?);
        self.trained = true;

        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let std_score = {
            let variance = cv_scores.iter()
                .map(|&x| (x - mean_score).powi(2))
                .sum::<f32>() / cv_scores.len() as f32;
            variance.sqrt()
        };

        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), std_score);
        results.insert("n_patterns".to_string(), self.pattern_names.len() as f32);

        Ok(results)
    }

    /// Make ensemble prediction with pattern contributions
    fn predict_pattern_ensemble(
        &self,
        py: Python,
        pattern_features: PyReadonlyArray1<f32>,
        _price_features: PyReadonlyArray1<f32>,
    ) -> PyResult<(i32, f32, Py<PyArray1<f32>>)> {
        if !self.trained {
            return Err(PyValueError::new_err("Model not trained"));
        }

        let pattern_array = pattern_features.as_array();
        let patterns = extract_safe!(pattern_array, "pattern_features");
        let weights = self.pattern_weights.as_ref().unwrap();

        let mut weighted_signals = Vec::new();
        let mut total_confidence = 0.0;
        let mut active_patterns = 0;

        // Calculate weighted ensemble prediction
        for (i, &pattern_signal) in patterns.iter().enumerate() {
            if i < self.pattern_names.len() && pattern_signal > 0.1 {
                let pattern_name = &self.pattern_names[i];
                if let Some(&weight) = weights.get(pattern_name) {
                    weighted_signals.push(pattern_signal * weight);
                    total_confidence += weight * pattern_signal;
                    active_patterns += 1;
                }
            }
        }

        let pattern_signal = if !weighted_signals.is_empty() {
            weighted_signals.iter().sum::<f32>() / weighted_signals.len() as f32
        } else { 0.0 };

        let prediction = if pattern_signal > 0.15 {
            2 // Buy
        } else if pattern_signal < -0.15 {
            0 // Sell
        } else {
            1 // Hold
        };

        let confidence = if active_patterns > 0 {
            (total_confidence / active_patterns as f32).min(1.0)
        } else { 0.0 };

        // Return pattern contributions for interpretability
        let mut pattern_contributions = vec![0.0f32; self.pattern_names.len()];
        for (i, pattern_name) in self.pattern_names.iter().enumerate() {
            if i < patterns.len() && patterns[i] > 0.1 {
                if let Some(&weight) = weights.get(pattern_name) {
                    pattern_contributions[i] = patterns[i] * weight;
                }
            }
        }

        Ok((
            prediction,
            confidence,
            PyArray1::from_vec(py, pattern_contributions).into(),
        ))
    }

    /// Get pattern importance scores
    fn get_pattern_importance(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        let importance_vec: Vec<f32> = self.pattern_names.iter()
            .map(|name| self.pattern_importance.get(name).unwrap_or(&0.0).clone())
            .collect();

        Ok(PyArray1::from_vec(py, importance_vec).into())
    }

    /// Get pattern names
    fn get_pattern_names(&self) -> Vec<String> {
        self.pattern_names.clone()
    }

    /// Set confidence threshold
    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        let _ = self.prediction_engine.set_confidence_threshold(threshold);
    }

    /// Check if model is trained
    fn is_trained(&self) -> bool {
        self.trained
    }
}

// Implementation of internal methods
impl PatternClassifier {
    /// Evaluate a cross-validation fold
    fn evaluate_fold(
        &self,
        pattern_features: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        test_idx: &[usize],
    ) -> PyResult<f32> {
        let mut correct = 0;
        let mut total = 0;

        for &idx in test_idx {
            if idx < pattern_features.nrows() && idx < y.len() {
                let pattern_row: Vec<f32> = pattern_features.row(idx).to_vec();
                let (pred_class, confidence) = self.predict_sample(&pattern_row)?;

                if confidence > self.confidence_threshold {
                    if pred_class == y[idx] {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }

        Ok(if total > 0 { correct as f32 / total as f32 } else { 0.0 })
    }

    /// Make prediction for a single sample
    fn predict_sample(&self, pattern_features: &[f32]) -> PyResult<(i32, f32)> {
        let pattern_signal = pattern_features.iter().sum::<f32>() / pattern_features.len() as f32;
        let confidence = pattern_signal.abs().min(1.0);

        let prediction = if pattern_signal > 0.2 {
            2
        } else if pattern_signal < -0.2 {
            0
        } else {
            1
        };

        Ok((prediction, confidence))
    }

    /// Calculate pattern ensemble weights
    fn calculate_pattern_weights(&self) -> PyResult<HashMap<String, f32>> {
        let mut weights = HashMap::new();
        let total_importance: f32 = self.pattern_importance.values().sum();

        if total_importance > 0.0 {
            for (pattern_name, &importance) in &self.pattern_importance {
                weights.insert(pattern_name.clone(), importance / total_importance);
            }
        } else {
            let uniform_weight = 1.0 / self.pattern_names.len() as f32;
            for pattern_name in &self.pattern_names {
                weights.insert(pattern_name.clone(), uniform_weight);
            }
        }

        Ok(weights)
    }
}

// Implement ML traits
impl MLBackend for PatternClassifier {
    fn train_model<'py>(
        &mut self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<HashMap<String, f32>> {
        // Use features for both pattern and price features (simplified)
        // Use the same features array for both parameters (they represent the same data)
        // Create a simple wrapper that doesn't require the same features twice
        let features_array = features.as_array();
        let y_array = labels.as_array();
        let (n_samples, _n_features) = features_array.dim();

        if y_array.len() != n_samples {
            return Err(PyValueError::new_err("Feature and label count mismatch"));
        }

        // Initialize sample weights if not set
        if self.sample_weights.len() != n_samples {
            self.sample_weights = vec![1.0; n_samples];
        }

        // Create pattern-aware cross-validation splits
        self.cv_splits = self.cross_validator.create_default_pattern_splits(n_samples, Some(3))?;

        let mut cv_scores = Vec::new();

        // Cross-validation training
        for (_train_idx, test_idx) in &self.cv_splits {
            let fold_score = self.evaluate_fold(&features_array, &y_array, test_idx)?;
            cv_scores.push(fold_score);
        }

        // Create ensemble weights
        self.pattern_weights = Some(self.calculate_pattern_weights()?);
        self.trained = true;

        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let std_score = {
            let variance = cv_scores.iter()
                .map(|&x| (x - mean_score).powi(2))
                .sum::<f32>() / cv_scores.len() as f32;
            variance.sqrt()
        };

        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), std_score);
        results.insert("n_patterns".to_string(), self.pattern_names.len() as f32);

        Ok(results)
    }

    fn predict_with_confidence<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        let features_array = features.as_array();
        let feats = extract_safe!(features_array, "features");
        self.predict_sample(feats)
    }

    fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        self.get_pattern_importance(py)
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn reset_model(&mut self) {
        self.trained = false;
        self.pattern_weights = None;
        self.pattern_importance.clear();
        self.cv_splits.clear();
        self.sample_weights.clear();
    }
}

impl LabelGenerator for PatternClassifier {
    fn create_triple_barrier_labels<'py>(
        &self,
        _py: Python<'py>,
        _prices: PyReadonlyArray1<'py, f32>,
        _volatility: PyReadonlyArray1<'py, f32>,
        _profit_mult: f32,
        _stop_mult: f32,
        _max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        Err(PyValueError::new_err("PatternClassifier uses pattern-based labels"))
    }

    fn create_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        open_prices: PyReadonlyArray1<'py, f32>,
        high_prices: PyReadonlyArray1<'py, f32>,
        low_prices: PyReadonlyArray1<'py, f32>,
        close_prices: PyReadonlyArray1<'py, f32>,
        future_periods: usize,
        profit_threshold: f32,
        stop_threshold: f32,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.pattern_labeler.create_pattern_labels(
            py, open_prices, high_prices, low_prices, close_prices,
            future_periods, profit_threshold, stop_threshold
        )
    }

    fn calculate_sample_weights<'py>(
        &self,
        _py: Python<'py>,
        _returns: PyReadonlyArray1<'py, f32>,
        _volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        Err(PyValueError::new_err("Use calculate_pattern_sample_weights for pattern-based weighting"))
    }
}

impl CrossValidator for PatternClassifier {
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.cross_validator.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
    }

    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.cross_validator.create_pattern_aware_cv_splits(n_samples, n_splits, pattern_duration)
    }

    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool {
        self.cross_validator.validate_cv_splits(splits, min_train_size, min_test_size)
    }
}

impl Predictor for PatternClassifier {
    fn predict_single<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        self.predict_with_confidence(py, features)
    }

    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
        self.prediction_engine.predict_batch(py, features)
    }

    fn predict_probabilities<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.prediction_engine.predict_probabilities(py, features)
    }

    fn get_prediction_explanation<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.prediction_engine.get_prediction_explanation(py, features)
    }

    fn set_confidence_threshold_unchecked(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
        self.prediction_engine.set_confidence_threshold_unchecked(threshold);
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

// Ensure thread safety
unsafe impl Send for PatternClassifier {}
unsafe impl Sync for PatternClassifier {}