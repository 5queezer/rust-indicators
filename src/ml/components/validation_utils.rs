//! # Validation Utilities
//!
//! Shared validation logic and utilities for ML models to eliminate code duplication
//! and provide consistent validation patterns across all classifiers.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::ndarray;
use std::collections::HashMap;

use crate::ml::components::phase4_integration::{Phase4Results, Phase4Workflow};

/// Common validation patterns and utilities shared across ML models
#[derive(Clone)]
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate feature and label dimensions match
    pub fn validate_dimensions(
        n_features: usize,
        n_labels: usize,
        context: &str,
    ) -> PyResult<()> {
        if n_features != n_labels {
            return Err(PyValueError::new_err(format!(
                "{}: Feature count ({}) and label count ({}) mismatch",
                context, n_features, n_labels
            )));
        }
        Ok(())
    }

    /// Validate minimum sample requirements
    pub fn validate_sample_count(
        n_samples: usize,
        min_samples: usize,
        context: &str,
    ) -> PyResult<()> {
        if n_samples < min_samples {
            return Err(PyValueError::new_err(format!(
                "{}: Insufficient samples ({} < {})",
                context, n_samples, min_samples
            )));
        }
        Ok(())
    }

    /// Create standardized cross-validation results
    pub fn create_cv_results(
        cv_scores: &[f32],
        additional_metrics: Option<HashMap<String, f32>>,
    ) -> HashMap<String, f32> {
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
        results.insert("n_splits".to_string(), cv_scores.len() as f32);

        // Add any additional metrics
        if let Some(metrics) = additional_metrics {
            results.extend(metrics);
        }

        results
    }

    /// Execute Phase 4 validation with common error handling
    pub fn execute_phase4_validation<F>(
        workflow: &mut Phase4Workflow,
        n_samples: usize,
        evaluate_fn: F,
    ) -> PyResult<Phase4Results>
    where
        F: Fn(&[usize], &[usize], usize) -> PyResult<(f32, f32)>,
    {
        workflow.execute_validation(n_samples, evaluate_fn)
            .map_err(|e| PyValueError::new_err(format!("Phase 4 validation failed: {}", e)))
    }

    /// Convert Phase4Results to standardized HashMap
    pub fn phase4_results_to_map(results: &Phase4Results) -> HashMap<String, f32> {
        let mut result_map = HashMap::new();
        result_map.insert("cv_mean".to_string(), results.cv_mean as f32);
        result_map.insert("cv_std".to_string(), results.cv_std as f32);
        result_map.insert("n_splits".to_string(), results.n_splits as f32);

        if let Some(pbo_result) = &results.pbo_result {
            result_map.insert("pbo_value".to_string(), pbo_result.pbo_value as f32);
            result_map.insert("is_overfit".to_string(), if pbo_result.is_overfit { 1.0 } else { 0.0 });
            result_map.insert("statistical_significance".to_string(), pbo_result.statistical_significance as f32);
            result_map.insert("confidence_lower".to_string(), pbo_result.confidence_interval.0 as f32);
            result_map.insert("confidence_upper".to_string(), pbo_result.confidence_interval.1 as f32);
            result_map.insert("n_combinations".to_string(), pbo_result.n_combinations as f32);
        }

        result_map
    }

    /// Common fold evaluation logic
    pub fn evaluate_fold_accuracy(
        predictions: &[(i32, f32)],
        actual_labels: &[i32],
        confidence_threshold: f32,
    ) -> f32 {
        let mut correct = 0;
        let mut total = 0;

        for (i, &(pred_class, confidence)) in predictions.iter().enumerate() {
            if i < actual_labels.len() && confidence > confidence_threshold {
                if pred_class == actual_labels[i] {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 { correct as f32 / total as f32 } else { 0.0 }
    }

    /// Initialize sample weights with validation
    pub fn initialize_sample_weights(
        n_samples: usize,
        existing_weights: &mut Vec<f32>,
    ) -> PyResult<()> {
        if existing_weights.len() != n_samples {
            *existing_weights = vec![1.0; n_samples];
        }
        Ok(())
    }

    /// Validate confidence threshold
    pub fn validate_confidence_threshold(threshold: f32) -> PyResult<f32> {
        let clamped = threshold.clamp(0.0, 1.0);
        if (threshold - clamped).abs() > f32::EPSILON {
            eprintln!("Warning: Confidence threshold {} clamped to {}", threshold, clamped);
        }
        Ok(clamped)
    }

    /// Common prediction validation
    pub fn validate_prediction_input(
        features: &ndarray::ArrayView1<f32>,
        expected_features: Option<usize>,
    ) -> PyResult<()> {
        if features.is_empty() {
            return Err(PyValueError::new_err("Empty feature array provided"));
        }

        if let Some(expected) = expected_features {
            if features.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "Feature count mismatch: expected {}, got {}",
                    expected, features.len()
                )));
            }
        }

        Ok(())
    }

    /// Common batch prediction validation
    pub fn validate_batch_prediction_input(
        features: &ndarray::ArrayView2<f32>,
        expected_features: Option<usize>,
    ) -> PyResult<()> {
        let (n_samples, n_features) = features.dim();
        
        if n_samples == 0 {
            return Err(PyValueError::new_err("Empty batch provided"));
        }

        if let Some(expected) = expected_features {
            if n_features != expected {
                return Err(PyValueError::new_err(format!(
                    "Feature count mismatch: expected {}, got {}",
                    expected, n_features
                )));
            }
        }

        Ok(())
    }
}

/// Shared evaluation functions for different model types
pub struct ModelEvaluators;

impl ModelEvaluators {
    /// Generic fold evaluation for pattern-based models
    pub fn evaluate_pattern_fold<F>(
        features: &ndarray::ArrayView2<f32>,
        labels: &ndarray::ArrayView1<i32>,
        test_indices: &[usize],
        predict_fn: F,
        confidence_threshold: f32,
    ) -> PyResult<f32>
    where
        F: Fn(&[f32]) -> PyResult<(i32, f32)>,
    {
        let mut predictions = Vec::new();
        let mut actual_labels = Vec::new();

        for &idx in test_indices {
            if idx < features.nrows() && idx < labels.len() {
                let feature_row: Vec<f32> = features.row(idx).to_vec();
                let prediction = predict_fn(&feature_row)?;
                predictions.push(prediction);
                actual_labels.push(labels[idx]);
            }
        }

        Ok(ValidationUtils::evaluate_fold_accuracy(
            &predictions,
            &actual_labels,
            confidence_threshold,
        ))
    }

    /// Generic fold evaluation for trading models
    pub fn evaluate_trading_fold<F>(
        features: &ndarray::ArrayView2<f32>,
        labels: &ndarray::ArrayView1<i32>,
        test_indices: &[usize],
        predict_fn: F,
        confidence_threshold: f32,
    ) -> PyResult<f32>
    where
        F: Fn(&[f32]) -> PyResult<(i32, f32)>,
    {
        // Same implementation as pattern fold for now, but can be specialized
        Self::evaluate_pattern_fold(features, labels, test_indices, predict_fn, confidence_threshold)
    }

    /// Generic fold evaluation for unified models
    pub fn evaluate_unified_fold<F>(
        features: &ndarray::ArrayView2<f32>,
        labels: &ndarray::ArrayView1<i32>,
        test_indices: &[usize],
        predict_fn: F,
        confidence_threshold: f32,
    ) -> PyResult<f32>
    where
        F: Fn(&[f32]) -> PyResult<(i32, f32)>,
    {
        // Same implementation as pattern fold for now, but can be specialized
        Self::evaluate_pattern_fold(features, labels, test_indices, predict_fn, confidence_threshold)
    }
}

/// Common training workflow patterns
pub struct TrainingWorkflows;

impl TrainingWorkflows {
    /// Standard cross-validation training workflow
    pub fn standard_cv_workflow<F, P>(
        features: &ndarray::ArrayView2<f32>,
        labels: &ndarray::ArrayView1<i32>,
        cv_splits: &[(Vec<usize>, Vec<usize>)],
        predict_fn: P,
        confidence_threshold: f32,
        fold_evaluator: F,
    ) -> PyResult<Vec<f32>>
    where
        F: Fn(&ndarray::ArrayView2<f32>, &ndarray::ArrayView1<i32>, &[usize], P, f32) -> PyResult<f32>,
        P: Fn(&[f32]) -> PyResult<(i32, f32)> + Copy,
    {
        let mut cv_scores = Vec::new();

        for (_train_idx, test_idx) in cv_splits {
            let fold_score = fold_evaluator(
                features,
                labels,
                test_idx,
                predict_fn,
                confidence_threshold,
            )?;
            cv_scores.push(fold_score);
        }

        Ok(cv_scores)
    }

    /// Phase 4 enhanced training workflow
    pub fn phase4_enhanced_workflow<F>(
        workflow: &mut Phase4Workflow,
        features: &ndarray::ArrayView2<f32>,
        labels: &ndarray::ArrayView1<i32>,
        evaluate_fn: F,
    ) -> PyResult<Phase4Results>
    where
        F: Fn(&ndarray::ArrayView2<f32>, &ndarray::ArrayView1<i32>, &[usize]) -> PyResult<f32>,
    {
        let (n_samples, _) = features.dim();
        
        let validation_fn = |_train_idx: &[usize], test_idx: &[usize], _combo_id: usize| -> PyResult<(f32, f32)> {
            let score = evaluate_fn(features, labels, test_idx)?;
            Ok((score, score)) // Return (train_score, test_score)
        };

        ValidationUtils::execute_phase4_validation(workflow, n_samples, validation_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_utils_dimensions() {
        assert!(ValidationUtils::validate_dimensions(100, 100, "test").is_ok());
        assert!(ValidationUtils::validate_dimensions(100, 50, "test").is_err());
    }

    #[test]
    fn test_validation_utils_sample_count() {
        assert!(ValidationUtils::validate_sample_count(100, 50, "test").is_ok());
        assert!(ValidationUtils::validate_sample_count(30, 50, "test").is_err());
    }

    #[test]
    fn test_cv_results_creation() {
        let scores = vec![0.8, 0.85, 0.75, 0.9];
        let results = ValidationUtils::create_cv_results(&scores, None);
        
        assert!(results.contains_key("cv_mean"));
        assert!(results.contains_key("cv_std"));
        assert!(results.contains_key("n_splits"));
        assert_eq!(results["n_splits"], 4.0);
    }

    #[test]
    fn test_confidence_threshold_validation() {
        assert_eq!(ValidationUtils::validate_confidence_threshold(0.5).unwrap(), 0.5);
        assert_eq!(ValidationUtils::validate_confidence_threshold(-0.1).unwrap(), 0.0);
        assert_eq!(ValidationUtils::validate_confidence_threshold(1.5).unwrap(), 1.0);
    }

    #[test]
    fn test_fold_accuracy_evaluation() {
        let predictions = vec![(1, 0.8), (0, 0.9), (1, 0.6), (2, 0.7)];
        let actual = vec![1, 0, 1, 1];
        let accuracy = ValidationUtils::evaluate_fold_accuracy(&predictions, &actual, 0.5);
        assert_eq!(accuracy, 0.75); // 3 out of 4 correct
    }
}