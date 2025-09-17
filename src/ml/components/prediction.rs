//! Prediction components with confidence scoring and batch processing
//!
//! This module provides shared prediction functionality extracted from both
//! pattern_model_example.rs and classifier_model_example.rs. It implements confidence
//! scoring, batch prediction, and model interpretability using the extract_safe! macro.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use crate::extract_safe;
use crate::ml::traits::Predictor;
use std::collections::HashMap;

/// Confidence-based prediction engine
///
/// This struct implements confidence scoring for predictions as used in both
/// example models, providing threshold-based filtering and interpretability.
#[derive(Debug, Clone)]
pub struct ConfidencePredictor {
    /// Confidence threshold for making predictions
    pub confidence_threshold: f32,
    /// Model weights for linear combination
    pub model_weights: Vec<f32>,
    /// Feature importance scores
    pub feature_importance: Vec<f32>,
    /// Whether the model is trained
    pub trained: bool,
}

impl ConfidencePredictor {
    /// Create a new confidence predictor
    ///
    /// # Parameters
    /// - `confidence_threshold`: Minimum confidence for predictions
    /// - `n_features`: Number of input features
    pub fn new(confidence_threshold: f32, n_features: usize) -> Self {
        Self {
            confidence_threshold,
            model_weights: vec![0.0; n_features],
            feature_importance: vec![0.0; n_features],
            trained: false,
        }
    }

    /// Create default confidence predictor
    pub fn default() -> Self {
        Self::new(0.6, 10)
    }

    /// Set model weights after training
    ///
    /// # Parameters
    /// - `weights`: Trained model weights
    /// - `importance`: Feature importance scores
    pub fn set_weights(&mut self, weights: Vec<f32>, importance: Vec<f32>) -> PyResult<()> {
        if weights.len() != importance.len() {
            return Err(PyValueError::new_err("Weights and importance must have same length"));
        }
        
        self.model_weights = weights;
        self.feature_importance = importance;
        self.trained = true;
        Ok(())
    }

    /// Make prediction for a single sample with confidence
    ///
    /// # Algorithm
    /// 1. Calculate weighted sum of features
    /// 2. Apply tanh activation for normalization
    /// 3. Calculate confidence as absolute value
    /// 4. Apply threshold filtering
    /// 5. Map to class prediction (0=sell, 1=hold, 2=buy)
    pub fn predict_sample(&self, features: &[f32]) -> Result<(i32, f32), Box<dyn std::error::Error>> {
        if !self.trained {
            return Err("Model not trained".into());
        }

        if features.len() != self.model_weights.len() {
            return Err("Feature dimension mismatch".into());
        }

        let weighted_sum = features.iter()
            .zip(&self.model_weights)
            .map(|(f, w)| f * w)
            .sum::<f32>();

        let normalized = weighted_sum.tanh(); // Squash to [-1, 1]
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

    /// Calculate feature contributions for interpretability
    ///
    /// # Parameters
    /// - `features`: Input feature vector
    ///
    /// # Returns
    /// Vector of feature contributions to the prediction
    pub fn get_feature_contributions(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !self.trained {
            return Err("Model not trained".into());
        }

        if features.len() != self.model_weights.len() {
            return Err("Feature dimension mismatch".into());
        }

        let contributions: Vec<f32> = features.iter()
            .zip(&self.model_weights)
            .zip(&self.feature_importance)
            .map(|((f, w), i)| f * w * i)
            .collect();

        Ok(contributions)
    }
}

impl Predictor for ConfidencePredictor {
    fn predict_single<'py>(
        &self,
        _py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        let features_array = features.as_array();
        let feats = extract_safe!(features_array, "features");
        
        let (prediction, confidence) = self.predict_sample(feats)
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
            
        Ok((prediction, confidence))
    }

    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
        if !self.trained {
            return Err(PyValueError::new_err("Model not trained"));
        }

        let features_array = features.as_array();
        let n_samples = features_array.nrows();
        
        let mut predictions = Vec::with_capacity(n_samples);
        let mut confidences = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let feature_row: Vec<f32> = features_array.row(i).to_vec();
            let (pred, conf) = self.predict_sample(&feature_row)
                .map_err(|e| PyValueError::new_err(format!("Batch prediction failed at sample {}: {}", i, e)))?;
            
            predictions.push(pred);
            confidences.push(conf);
        }

        Ok((
            PyArray1::from_vec(py, predictions).to_owned().into(),
            PyArray1::from_vec(py, confidences).to_owned().into(),
        ))
    }

    fn predict_probabilities<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let features_array = features.as_array();
        let feats = extract_safe!(features_array, "features");
        
        let (prediction, confidence) = self.predict_sample(feats)
            .map_err(|e| PyValueError::new_err(format!("Probability prediction failed: {}", e)))?;

        // Convert to class probabilities
        let mut probs = vec![0.0f32; 3]; // [sell, hold, buy]
        
        // Distribute confidence based on prediction
        match prediction {
            0 => probs[0] = confidence, // Sell
            1 => probs[1] = confidence, // Hold
            2 => probs[2] = confidence, // Buy
            _ => return Err(PyValueError::new_err("Invalid prediction class")),
        }
        
        // Normalize remaining probability mass
        let remaining = 1.0 - confidence;
        let other_prob = remaining / 2.0;
        
        for i in 0..3 {
            if i != prediction as usize {
                probs[i] = other_prob;
            }
        }

        Ok(PyArray1::from_vec(py, probs).to_owned().into())
    }

    fn get_prediction_explanation<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let features_array = features.as_array();
        let feats = extract_safe!(features_array, "features");
        
        let contributions = self.get_feature_contributions(feats)
            .map_err(|e| PyValueError::new_err(format!("Feature contribution calculation failed: {}", e)))?;

        Ok(PyArray1::from_vec(py, contributions).to_owned().into())
    }

    fn set_confidence_threshold_unchecked(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

/// Batch prediction engine for efficient processing
///
/// This struct provides optimized batch prediction capabilities
/// for backtesting and bulk processing scenarios.
#[derive(Debug, Clone)]
pub struct BatchPredictor {
    /// Base confidence predictor
    pub base_predictor: ConfidencePredictor,
    /// Batch size for processing
    pub batch_size: usize,
}

impl BatchPredictor {
    /// Create a new batch predictor
    ///
    /// # Parameters
    /// - `base_predictor`: Underlying confidence predictor
    /// - `batch_size`: Size of processing batches
    pub fn new(base_predictor: ConfidencePredictor, batch_size: usize) -> Self {
        Self {
            base_predictor,
            batch_size,
        }
    }

    /// Create default batch predictor
    pub fn default() -> Self {
        Self::new(ConfidencePredictor::default(), 1000)
    }

    /// Process large datasets in batches
    ///
    /// # Parameters
    /// - `py`: Python context
    /// - `features`: Large feature matrix
    ///
    /// # Returns
    /// Tuple of (predictions, confidences) for all samples
    pub fn predict_large_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
        let features_array = features.as_array();
        let n_samples = features_array.nrows();
        
        let mut all_predictions = Vec::with_capacity(n_samples);
        let mut all_confidences = Vec::with_capacity(n_samples);

        // Process in batches to manage memory
        for start_idx in (0..n_samples).step_by(self.batch_size) {
            let end_idx = (start_idx + self.batch_size).min(n_samples);
            
            for i in start_idx..end_idx {
                let feature_row: Vec<f32> = features_array.row(i).to_vec();
                let (pred, conf) = self.base_predictor.predict_sample(&feature_row)
                    .map_err(|e| PyValueError::new_err(format!("Batch prediction failed at sample {}: {}", i, e)))?;
                
                all_predictions.push(pred);
                all_confidences.push(conf);
            }
        }

        Ok((
            PyArray1::from_vec(py, all_predictions).to_owned().into(),
            PyArray1::from_vec(py, all_confidences).to_owned().into(),
        ))
    }
}

/// Unified prediction engine combining all prediction capabilities
///
/// This struct provides a complete prediction interface implementing
/// all prediction-related functionality from the example models.
#[derive(Debug, Clone)]
pub struct PredictionEngine {
    /// Confidence predictor
    pub confidence_predictor: ConfidencePredictor,
    /// Batch predictor
    pub batch_predictor: BatchPredictor,
    /// Pattern-specific prediction parameters
    pub pattern_params: HashMap<String, f32>,
}

impl PredictionEngine {
    /// Create a new prediction engine
    pub fn new(
        confidence_predictor: ConfidencePredictor,
        batch_predictor: BatchPredictor,
    ) -> Self {
        Self {
            confidence_predictor,
            batch_predictor,
            pattern_params: HashMap::new(),
        }
    }

    /// Create default prediction engine
    pub fn default() -> Self {
        let confidence_predictor = ConfidencePredictor::default();
        let batch_predictor = BatchPredictor::new(confidence_predictor.clone(), 1000);
        
        Self::new(confidence_predictor, batch_predictor)
    }

    /// Set pattern-specific parameters
    ///
    /// # Parameters
    /// - `pattern_name`: Name of the pattern
    /// - `weight`: Weight for the pattern
    pub fn set_pattern_weight(&mut self, pattern_name: String, weight: f32) {
        self.pattern_params.insert(pattern_name, weight);
    }

    /// Get pattern weights
    pub fn get_pattern_weights(&self) -> &HashMap<String, f32> {
        &self.pattern_params
    }
}

impl Predictor for PredictionEngine {
    fn predict_single<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        self.confidence_predictor.predict_single(py, features)
    }

    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>)> {
        self.batch_predictor.predict_large_batch(py, features)
    }

    fn predict_probabilities<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.confidence_predictor.predict_probabilities(py, features)
    }

    fn get_prediction_explanation<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.confidence_predictor.get_prediction_explanation(py, features)
    }

    fn set_confidence_threshold_unchecked(&mut self, threshold: f32) {
        self.confidence_predictor.set_confidence_threshold_unchecked(threshold);
        self.batch_predictor.base_predictor.set_confidence_threshold_unchecked(threshold);
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.confidence_predictor.get_confidence_threshold()
    }
}

// Ensure thread safety
unsafe impl Send for ConfidencePredictor {}
unsafe impl Sync for ConfidencePredictor {}
unsafe impl Send for BatchPredictor {}
unsafe impl Sync for BatchPredictor {}
unsafe impl Send for PredictionEngine {}
unsafe impl Sync for PredictionEngine {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_predictor_creation() {
        let predictor = ConfidencePredictor::new(0.6, 5);
        assert_eq!(predictor.confidence_threshold, 0.6);
        assert_eq!(predictor.model_weights.len(), 5);
        assert!(!predictor.trained);
    }

    #[test]
    fn test_batch_predictor_creation() {
        let base = ConfidencePredictor::default();
        let batch = BatchPredictor::new(base, 100);
        assert_eq!(batch.batch_size, 100);
    }

    #[test]
    fn test_prediction_engine_creation() {
        let engine = PredictionEngine::default();
        assert_eq!(engine.confidence_predictor.confidence_threshold, 0.6);
        assert_eq!(engine.batch_predictor.batch_size, 1000);
    }

    #[test]
    fn test_pattern_weights() {
        let mut engine = PredictionEngine::default();
        engine.set_pattern_weight("test_pattern".to_string(), 0.8);
        
        let weights = engine.get_pattern_weights();
        assert_eq!(weights.get("test_pattern"), Some(&0.8));
    }
}