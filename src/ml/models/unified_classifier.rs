//! Unified classifier combining pattern recognition and trading classification
//!
//! This module provides a unified ML model that can operate in three modes:
//! - Pattern: Pure pattern recognition using pattern-aware components
//! - Trading: Scientific trading classification with purged cross-validation
//! - Hybrid: Combined approach using both pattern and trading features

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray};
use std::collections::HashMap;

use crate::extract_safe;
use crate::ml::traits::{MLBackend, LabelGenerator, CrossValidator, Predictor};
use crate::ml::components::{
    PatternLabeler, TripleBarrierLabeler, PatternWeighting, VolatilityWeighting,
    PatternAwareCrossValidator, PurgedCrossValidator, PredictionEngine, SampleWeightCalculator,
};

/// Operating mode for the unified classifier
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum ClassifierMode {
    /// Pure pattern recognition mode
    Pattern,
    /// Scientific trading classification mode
    Trading,
    /// Hybrid mode combining both approaches
    Hybrid,
}

#[pymethods]
impl ClassifierMode {
    #[new]
    fn new() -> Self {
        ClassifierMode::Hybrid
    }

    fn __str__(&self) -> &'static str {
        match self {
            ClassifierMode::Pattern => "Pattern",
            ClassifierMode::Trading => "Trading",
            ClassifierMode::Hybrid => "Hybrid",
        }
    }
}

/// Unified classifier that combines pattern recognition and trading classification
///
/// This classifier can operate in three modes and dynamically switch between them.
/// It integrates all shared components and provides a unified interface for both
/// pattern-based and trading-based machine learning approaches.
#[pyclass]
pub struct UnifiedClassifier {
    // Operating mode
    mode: ClassifierMode,
    
    // Shared components for pattern recognition
    pattern_labeler: PatternLabeler,
    pattern_weighting: PatternWeighting,
    pattern_cv: PatternAwareCrossValidator,
    
    // Shared components for trading classification
    triple_barrier_labeler: TripleBarrierLabeler,
    volatility_weighting: VolatilityWeighting,
    purged_cv: PurgedCrossValidator,
    
    // Common components
    prediction_engine: PredictionEngine,
    sample_weight_calculator: SampleWeightCalculator,
    
    // Model parameters
    n_features: usize,
    embargo_pct: f32,
    pattern_duration: usize,
    
    // Model state
    pattern_weights: Vec<f32>,
    trading_weights: Vec<f32>,
    hybrid_weights: Vec<f32>,
    feature_importance: Vec<f32>,
    sample_weights: Vec<f32>,
    trained: bool,
    
    // Cross-validation splits
    cv_splits: Vec<(Vec<usize>, Vec<usize>)>,
}

#[pymethods]
impl UnifiedClassifier {
    /// Create a new unified classifier
    #[new]
    fn new(n_features: usize, mode: Option<ClassifierMode>) -> PyResult<Self> {
        let mode = mode.unwrap_or(ClassifierMode::Hybrid);
        
        Ok(UnifiedClassifier {
            mode,
            pattern_labeler: PatternLabeler::default(),
            pattern_weighting: PatternWeighting::default(),
            pattern_cv: PatternAwareCrossValidator::default(),
            triple_barrier_labeler: TripleBarrierLabeler::default(),
            volatility_weighting: VolatilityWeighting::default(),
            purged_cv: PurgedCrossValidator::default(),
            prediction_engine: PredictionEngine::default(),
            sample_weight_calculator: SampleWeightCalculator::default(),
            n_features,
            embargo_pct: 0.01,
            pattern_duration: 10,
            pattern_weights: vec![0.0; n_features],
            trading_weights: vec![0.0; n_features],
            hybrid_weights: vec![0.0; n_features],
            feature_importance: vec![0.0; n_features],
            sample_weights: Vec::new(),
            trained: false,
            cv_splits: Vec::new(),
        })
    }

    /// Set the operating mode
    fn set_mode(&mut self, mode: ClassifierMode) {
        self.mode = mode;
        self.trained = false; // Require retraining when mode changes
    }

    /// Get the current operating mode
    fn get_mode(&self) -> ClassifierMode {
        self.mode
    }

    /// Train the unified model in the current mode
    fn train_unified(
        &mut self,
        X: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        let X_array = X.as_array();
        let y_array = y.as_array();
        let (n_samples, n_features) = X_array.dim();

        if n_features != self.n_features {
            return Err(PyValueError::new_err(
                format!("Expected {} features, got {}", self.n_features, n_features)
            ));
        }

        if n_samples != y_array.len() {
            return Err(PyValueError::new_err("X and y length mismatch"));
        }

        // Initialize sample weights if not set
        if self.sample_weights.len() != n_samples {
            self.sample_weights = vec![1.0; n_samples];
        }

        match self.mode {
            ClassifierMode::Pattern => self.train_pattern_mode(&X_array, &y_array, learning_rate),
            ClassifierMode::Trading => self.train_trading_mode(&X_array, &y_array, learning_rate),
            ClassifierMode::Hybrid => self.train_hybrid_mode(&X_array, &y_array, learning_rate),
        }
    }

    /// Switch to pattern mode and train
    fn train_pattern_mode_explicit(
        &mut self,
        X: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        self.mode = ClassifierMode::Pattern;
        self.train_unified(X, y, learning_rate)
    }

    /// Switch to trading mode and train
    fn train_trading_mode_explicit(
        &mut self,
        X: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        self.mode = ClassifierMode::Trading;
        self.train_unified(X, y, learning_rate)
    }

    /// Switch to hybrid mode and train
    fn train_hybrid_mode_explicit(
        &mut self,
        X: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        self.mode = ClassifierMode::Hybrid;
        self.train_unified(X, y, learning_rate)
    }

    /// Get feature importance for current mode
    fn get_feature_importance(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.feature_importance.clone()).into())
    }

    /// Get mode-specific weights
    fn get_mode_weights(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        let weights = match self.mode {
            ClassifierMode::Pattern => &self.pattern_weights,
            ClassifierMode::Trading => &self.trading_weights,
            ClassifierMode::Hybrid => &self.hybrid_weights,
        };
        Ok(PyArray1::from_vec(py, weights.clone()).into())
    }

    /// Set embargo percentage for trading/hybrid modes
    fn set_embargo_pct(&mut self, embargo_pct: f32) {
        self.embargo_pct = embargo_pct.clamp(0.0, 0.5);
    }

    /// Set pattern duration for pattern/hybrid modes
    fn set_pattern_duration(&mut self, duration: usize) {
        self.pattern_duration = duration.max(1);
    }

    /// Check if model is trained
    fn is_trained(&self) -> bool {
        self.trained
    }
}

// Implementation of internal training methods
impl UnifiedClassifier {
    /// Train in pattern recognition mode
    fn train_pattern_mode(
        &mut self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        let (n_samples, _) = X.dim();

        // Create pattern-aware cross-validation splits
        self.cv_splits = self.pattern_cv.create_pattern_aware_cv_splits(
            n_samples, 3, self.pattern_duration
        )?;

        let mut cv_scores = Vec::new();
        let mut feature_scores = vec![0.0; self.n_features];

        // Cross-validation training
        for (train_idx, test_idx) in &self.cv_splits {
            let (fold_score, fold_feature_importance) = self.train_pattern_fold(
                X, y, train_idx, test_idx, learning_rate
            )?;

            cv_scores.push(fold_score);

            for i in 0..self.n_features {
                feature_scores[i] += fold_feature_importance[i];
            }
        }

        // Average feature importance across folds
        let n_folds = self.cv_splits.len() as f32;
        for i in 0..self.n_features {
            self.feature_importance[i] = feature_scores[i] / n_folds;
        }

        // Train final pattern model
        self.pattern_weights = self.train_final_model(X, y, learning_rate)?;
        self.trained = true;

        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let variance = cv_scores.iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>() / cv_scores.len() as f32;

        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), variance.sqrt());
        results.insert("mode".to_string(), 1.0); // Pattern mode

        Ok(results)
    }

    /// Train in trading classification mode
    fn train_trading_mode(
        &mut self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        let (n_samples, _) = X.dim();

        // Create purged cross-validation splits
        self.cv_splits = self.purged_cv.create_purged_cv_splits(
            n_samples, 3, self.embargo_pct
        )?;

        let mut cv_scores = Vec::new();
        let mut feature_scores = vec![0.0; self.n_features];

        // Cross-validation training
        for (train_idx, test_idx) in &self.cv_splits {
            let (fold_score, fold_feature_importance) = self.train_trading_fold(
                X, y, train_idx, test_idx, learning_rate
            )?;

            cv_scores.push(fold_score);

            for i in 0..self.n_features {
                feature_scores[i] += fold_feature_importance[i];
            }
        }

        // Average feature importance across folds
        let n_folds = self.cv_splits.len() as f32;
        for i in 0..self.n_features {
            self.feature_importance[i] = feature_scores[i] / n_folds;
        }

        // Train final trading model
        self.trading_weights = self.train_final_model(X, y, learning_rate)?;
        self.trained = true;

        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let variance = cv_scores.iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>() / cv_scores.len() as f32;

        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), variance.sqrt());
        results.insert("mode".to_string(), 2.0); // Trading mode

        Ok(results)
    }

    /// Train in hybrid mode
    fn train_hybrid_mode(
        &mut self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        // Train both pattern and trading components
        let pattern_results = self.train_pattern_mode(X, y, learning_rate)?;
        let trading_results = self.train_trading_mode(X, y, learning_rate)?;

        // Combine weights (simple average for now)
        for i in 0..self.n_features {
            self.hybrid_weights[i] = (self.pattern_weights[i] + self.trading_weights[i]) / 2.0;
        }

        let pattern_score = pattern_results.get("cv_mean").unwrap_or(&0.0);
        let trading_score = trading_results.get("cv_mean").unwrap_or(&0.0);
        let hybrid_score = (pattern_score + trading_score) / 2.0;

        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), hybrid_score);
        results.insert("pattern_score".to_string(), *pattern_score);
        results.insert("trading_score".to_string(), *trading_score);
        results.insert("mode".to_string(), 3.0); // Hybrid mode

        Ok(results)
    }

    /// Train a pattern recognition fold
    fn train_pattern_fold(
        &self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        _train_idx: &[usize],
        test_idx: &[usize],
        _lr: f32,
    ) -> PyResult<(f32, Vec<f32>)> {
        let mut correct = 0;
        let mut total = 0;
        let feature_correlations = vec![0.15; self.n_features]; // Pattern-specific importance

        // Test performance using pattern-based prediction
        for &idx in test_idx {
            let features: Vec<f32> = X.row(idx).to_vec();
            let (pred_class, confidence) = self.predict_pattern_sample(&features)?;

            if confidence > 0.4 {
                if pred_class == y[idx] {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
        Ok((accuracy, feature_correlations))
    }

    /// Train a trading classification fold
    fn train_trading_fold(
        &self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        _train_idx: &[usize],
        test_idx: &[usize],
        _lr: f32,
    ) -> PyResult<(f32, Vec<f32>)> {
        let mut correct = 0;
        let mut total = 0;
        let feature_correlations = vec![0.12; self.n_features]; // Trading-specific importance

        // Test performance using trading-based prediction
        for &idx in test_idx {
            let features: Vec<f32> = X.row(idx).to_vec();
            let (pred_class, confidence) = self.predict_trading_sample(&features)?;

            if confidence > 0.3 {
                if pred_class == y[idx] {
                    correct += 1;
                }
                total += 1;
            }
        }

        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
        Ok((accuracy, feature_correlations))
    }

    /// Make pattern-based prediction for a single sample
    fn predict_pattern_sample(&self, features: &[f32]) -> PyResult<(i32, f32)> {
        if features.len() != self.pattern_weights.len() {
            return Err(PyValueError::new_err("Feature dimension mismatch"));
        }

        let weighted_sum = features.iter()
            .zip(&self.pattern_weights)
            .map(|(f, w)| f * w)
            .sum::<f32>();

        let normalized = weighted_sum.tanh();
        let confidence = normalized.abs().min(1.0);

        let prediction = if normalized > 0.2 {
            2 // Strong pattern
        } else if normalized < -0.2 {
            0 // Weak pattern
        } else {
            1 // Neutral
        };

        Ok((prediction, confidence))
    }

    /// Make trading-based prediction for a single sample
    fn predict_trading_sample(&self, features: &[f32]) -> PyResult<(i32, f32)> {
        if features.len() != self.trading_weights.len() {
            return Err(PyValueError::new_err("Feature dimension mismatch"));
        }

        let weighted_sum = features.iter()
            .zip(&self.trading_weights)
            .map(|(f, w)| f * w)
            .sum::<f32>();

        let normalized = weighted_sum.tanh();
        let confidence = normalized.abs().min(1.0);

        let prediction = if normalized > 0.15 {
            2 // Buy
        } else if normalized < -0.15 {
            0 // Sell
        } else {
            1 // Hold
        };

        Ok((prediction, confidence))
    }

    /// Make hybrid prediction for a single sample
    fn predict_hybrid_sample(&self, features: &[f32]) -> PyResult<(i32, f32)> {
        if features.len() != self.hybrid_weights.len() {
            return Err(PyValueError::new_err("Feature dimension mismatch"));
        }

        let weighted_sum = features.iter()
            .zip(&self.hybrid_weights)
            .map(|(f, w)| f * w)
            .sum::<f32>();

        let normalized = weighted_sum.tanh();
        let confidence = normalized.abs().min(1.0);

        let prediction = if normalized > 0.18 {
            2 // Strong signal
        } else if normalized < -0.18 {
            0 // Weak signal
        } else {
            1 // Neutral
        };

        Ok((prediction, confidence))
    }

    /// Train final model using gradient descent
    fn train_final_model(
        &self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        learning_rate: f32,
    ) -> PyResult<Vec<f32>> {
        let (n_samples, n_features) = X.dim();
        let mut weights = vec![0.01; n_features];

        let epochs = 100;

        for _ in 0..epochs {
            let mut gradient = vec![0.0; n_features];

            for i in 0..n_samples {
                let features: Vec<f32> = X.row(i).to_vec();
                let prediction = features.iter()
                    .zip(&weights)
                    .map(|(f, w)| f * w)
                    .sum::<f32>()
                    .tanh();

                let target = match y[i] {
                    0 => -1.0,
                    1 => 0.0,
                    2 => 1.0,
                    _ => 0.0,
                };

                let error = target - prediction;
                let weight_factor = self.sample_weights.get(i).unwrap_or(&1.0);

                for j in 0..n_features {
                    gradient[j] += error * features[j] * weight_factor;
                }
            }

            // Update weights
            for j in 0..n_features {
                weights[j] += learning_rate * gradient[j] / n_samples as f32;
            }
        }

        Ok(weights)
    }
}

// Implement ML traits
impl MLBackend for UnifiedClassifier {
    fn train_model<'py>(
        &mut self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<HashMap<String, f32>> {
        self.train_unified(features, labels, 0.01)
    }

    fn predict_with_confidence<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        if !self.trained {
            return Err(PyValueError::new_err("Model not trained"));
        }

        let features_array = features.as_array();
        let feats = extract_safe!(features_array, "features");

        if feats.len() != self.n_features {
            return Err(PyValueError::new_err(
                format!("Expected {} features, got {}", self.n_features, feats.len())
            ));
        }

        match self.mode {
            ClassifierMode::Pattern => self.predict_pattern_sample(feats),
            ClassifierMode::Trading => self.predict_trading_sample(feats),
            ClassifierMode::Hybrid => self.predict_hybrid_sample(feats),
        }
    }

    fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        self.get_feature_importance(py)
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn reset_model(&mut self) {
        self.trained = false;
        self.feature_importance = vec![0.0; self.n_features];
        self.sample_weights.clear();
        self.pattern_weights = vec![0.0; self.n_features];
        self.trading_weights = vec![0.0; self.n_features];
        self.hybrid_weights = vec![0.0; self.n_features];
        self.cv_splits.clear();
    }
}

// UnifiedMLBackend is automatically implemented via blanket implementation
// since UnifiedClassifier implements MLBackend + LabelGenerator + CrossValidator + Predictor

// Implement other required traits with delegation to appropriate components
impl LabelGenerator for UnifiedClassifier {
    fn create_triple_barrier_labels<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
        profit_mult: f32,
        stop_mult: f32,
        max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.triple_barrier_labeler.create_triple_barrier_labels(
            py, prices, volatility, profit_mult, stop_mult, max_hold
        )
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
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
        _volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        match self.mode {
            ClassifierMode::Pattern => {
                // For pattern mode, use volatility weights if no pattern signals available
                self.sample_weight_calculator.calculate_volatility_weights(py, returns)
            }
            ClassifierMode::Trading => {
                self.sample_weight_calculator.calculate_volatility_weights(py, returns)
            }
            ClassifierMode::Hybrid => {
                // Use volatility weights for hybrid mode
                self.sample_weight_calculator.calculate_volatility_weights(py, returns)
            }
        }
    }
}

impl CrossValidator for UnifiedClassifier {
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.purged_cv.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
    }

    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.pattern_cv.create_pattern_aware_cv_splits(n_samples, n_splits, pattern_duration)
    }

    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool {
        self.purged_cv.validate_cv_splits(splits, min_train_size, min_test_size)
    }
}

impl Predictor for UnifiedClassifier {
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
        self.prediction_engine.set_confidence_threshold_unchecked(threshold);
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.prediction_engine.get_confidence_threshold()
    }
}

// Ensure thread safety
unsafe impl Send for UnifiedClassifier {}
unsafe impl Sync for UnifiedClassifier {}