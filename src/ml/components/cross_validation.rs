//! Cross-validation components with purged splits and embargo logic
//!
//! This module provides shared cross-validation functionality that prevents data leakage
//! in time series data through proper embargo periods and purged splits. It implements
//! the patterns from both pattern_model_example.rs and classifier_model_example.rs.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::ml::traits::CrossValidator;

/// Purged cross-validation implementation with embargo periods
///
/// This struct provides time-series aware cross-validation that prevents data leakage
/// by implementing embargo periods between training and test sets. It follows the
/// production-ready patterns from ScientificTradingClassifier.
///
/// # Thread Safety
/// 
/// This struct is `Send + Sync` to support multi-threaded access from Python.
#[derive(Debug, Clone)]
pub struct PurgedCrossValidator {
    /// Embargo percentage as fraction of total samples
    pub embargo_pct: f32,
    /// Minimum training set size
    pub min_train_size: usize,
    /// Minimum test set size  
    pub min_test_size: usize,
}

impl PurgedCrossValidator {
    /// Create a new purged cross-validator
    ///
    /// # Parameters
    /// - `embargo_pct`: Embargo period as percentage of total samples (e.g., 0.01 for 1%)
    /// - `min_train_size`: Minimum required training set size
    /// - `min_test_size`: Minimum required test set size
    ///
    /// # Returns
    /// New PurgedCrossValidator instance
    ///
    /// # Example
    /// ```rust,ignore
    /// let validator = PurgedCrossValidator::new(0.01, 100, 20);
    /// ```
    pub fn new(embargo_pct: f32, min_train_size: usize, min_test_size: usize) -> Self {
        Self {
            embargo_pct,
            min_train_size,
            min_test_size,
        }
    }

    /// Create default purged cross-validator with standard parameters
    ///
    /// Uses 1% embargo, minimum 50 training samples, minimum 10 test samples.
    pub fn default() -> Self {
        Self::new(0.01, 50, 10)
    }
}

impl CrossValidator for PurgedCrossValidator {
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_samples < n_splits {
            return Err(PyValueError::new_err("Not enough samples for splits"));
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / n_splits;
        let embargo = ((n_samples as f32 * embargo_pct) as usize).max(1);

        for fold in 0..n_splits {
            let test_start = fold * fold_size;
            let test_end = if fold == n_splits - 1 { n_samples } else { (fold + 1) * fold_size };

            if test_start >= test_end {
                continue;
            }

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let mut train_indices = Vec::new();

            // Add training data before test set (with embargo)
            if test_start > embargo {
                train_indices.extend(0..(test_start - embargo));
            }

            // Add training data after test set (with embargo)
            if test_end + embargo < n_samples {
                train_indices.extend((test_end + embargo)..n_samples);
            }

            // Only add split if it meets minimum size requirements
            if train_indices.len() >= self.min_train_size && test_indices.len() >= self.min_test_size {
                splits.push((train_indices, test_indices));
            }
        }

        if splits.is_empty() {
            return Err(PyValueError::new_err("No valid CV splits created"));
        }

        Ok(splits)
    }

    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        // Use pattern duration * 2 as minimum embargo to prevent pattern overlap
        let pattern_embargo = (pattern_duration * 2).max(10);
        let embargo_pct = (pattern_embargo as f32) / (n_samples as f32);
        
        self.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
    }

    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool {
        for (train_indices, test_indices) in splits {
            if train_indices.len() < min_train_size || test_indices.len() < min_test_size {
                return false;
            }

            // Check for overlap between train and test sets
            for &test_idx in test_indices {
                if train_indices.contains(&test_idx) {
                    return false;
                }
            }
        }
        true
    }
}

/// Pattern-aware cross-validation for pattern recognition models
///
/// This struct extends purged cross-validation with pattern-specific logic,
/// implementing the patterns from PatternRecognitionClassifier.
#[derive(Debug, Clone)]
pub struct PatternAwareCrossValidator {
    /// Base purged validator
    pub base_validator: PurgedCrossValidator,
    /// Default pattern duration in bars
    pub default_pattern_duration: usize,
}

impl PatternAwareCrossValidator {
    /// Create a new pattern-aware cross-validator
    ///
    /// # Parameters
    /// - `embargo_pct`: Base embargo percentage
    /// - `pattern_duration`: Default pattern duration in bars
    ///
    /// # Example
    /// ```rust,ignore
    /// let validator = PatternAwareCrossValidator::new(0.01, 5);
    /// ```
    pub fn new(embargo_pct: f32, pattern_duration: usize) -> Self {
        Self {
            base_validator: PurgedCrossValidator::new(embargo_pct, 50, 10),
            default_pattern_duration: pattern_duration,
        }
    }

    /// Create default pattern-aware validator
    ///
    /// Uses 5-bar pattern duration as default.
    pub fn default() -> Self {
        Self::new(0.01, 5)
    }

    /// Create pattern-aware CV splits with default pattern duration
    ///
    /// # Parameters
    /// - `n_samples`: Total number of samples
    /// - `n_splits`: Number of CV folds (default: 3)
    ///
    /// # Returns
    /// Vector of (train_indices, test_indices) tuples
    pub fn create_default_pattern_splits(
        &self,
        n_samples: usize,
        n_splits: Option<usize>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let splits = n_splits.unwrap_or(3);
        self.base_validator.create_pattern_aware_cv_splits(
            n_samples,
            splits,
            self.default_pattern_duration,
        )
    }
}

impl CrossValidator for PatternAwareCrossValidator {
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.base_validator.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
    }

    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        self.base_validator.create_pattern_aware_cv_splits(n_samples, n_splits, pattern_duration)
    }

    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool {
        self.base_validator.validate_cv_splits(splits, min_train_size, min_test_size)
    }
}

// Ensure thread safety
unsafe impl Send for PurgedCrossValidator {}
unsafe impl Sync for PurgedCrossValidator {}
unsafe impl Send for PatternAwareCrossValidator {}
unsafe impl Sync for PatternAwareCrossValidator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purged_cv_basic() {
        let validator = PurgedCrossValidator::new(0.1, 10, 5);
        let splits = validator.create_purged_cv_splits(100, 3, 0.1).unwrap();
        
        assert!(!splits.is_empty());
        assert!(splits.len() <= 3);
        
        // Verify no overlap between train and test
        for (train, test) in &splits {
            for &test_idx in test {
                assert!(!train.contains(&test_idx));
            }
        }
    }

    #[test]
    fn test_pattern_aware_cv() {
        let validator = PatternAwareCrossValidator::new(0.05, 5);
        let splits = validator.create_default_pattern_splits(100, Some(3)).unwrap();
        
        assert!(!splits.is_empty());
        
        // Verify splits are valid
        assert!(validator.validate_cv_splits(&splits, 10, 5));
    }

    #[test]
    fn test_insufficient_samples() {
        let validator = PurgedCrossValidator::new(0.1, 10, 5);
        let result = validator.create_purged_cv_splits(5, 3, 0.1);
        
        assert!(result.is_err());
    }
}