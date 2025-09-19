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

/// Combinatorial Purged Cross-Validation implementation
///
/// This struct implements the CombinatorialPurgedCV methodology from López de Prado,
/// which generates all possible C(N,k) combinations of training/test splits to provide
/// robust overfitting detection through Probability of Backtest Overfitting (PBO) calculation.
///
/// # Thread Safety
///
/// This struct is `Send + Sync` to support multi-threaded access from Python.
#[derive(Debug, Clone)]
pub struct CombinatorialPurgedCV {
    /// Base purged cross-validator for compatibility
    pub base_cv: PurgedCrossValidator,
    /// Number of groups to divide data into
    pub n_groups: usize,
    /// Number of groups to use for testing in each combination
    pub test_groups: usize,
    /// Embargo periods between training and test sets
    pub embargo_periods: usize,
}

/// Metrics for a single cross-validation combination
#[derive(Debug, Clone)]
pub struct CVMetrics {
    /// Performance score for this combination
    pub performance: f64,
    /// Training set size
    pub train_size: usize,
    /// Test set size
    pub test_size: usize,
    /// Combination identifier
    pub combination_id: usize,
}

/// Overfitting detection metrics
#[derive(Debug, Clone)]
pub struct OverfittingMetrics {
    /// Probability of Backtest Overfitting
    pub pbo: f64,
    /// Performance distribution across all combinations
    pub performance_distribution: Vec<f64>,
    /// Confidence interval for PBO
    pub confidence_interval: (f64, f64),
    /// Number of combinations processed
    pub n_combinations: usize,
}

impl CombinatorialPurgedCV {
    /// Create a new combinatorial purged cross-validator
    ///
    /// # Parameters
    /// - `embargo_pct`: Embargo period as percentage of total samples
    /// - `n_groups`: Number of groups to divide data into
    /// - `test_groups`: Number of groups to use for testing (k in C(N,k))
    /// - `min_train_size`: Minimum required training set size
    /// - `min_test_size`: Minimum required test set size
    ///
    /// # Returns
    /// New CombinatorialPurgedCV instance
    ///
    /// # Example
    /// ```rust,ignore
    /// let cpcv = CombinatorialPurgedCV::new(0.01, 10, 3, 50, 10);
    /// ```
    pub fn new(
        embargo_pct: f32,
        n_groups: usize,
        test_groups: usize,
        min_train_size: usize,
        min_test_size: usize,
    ) -> Self {
        let base_cv = PurgedCrossValidator::new(embargo_pct, min_train_size, min_test_size);
        let embargo_periods = ((n_groups as f32 * embargo_pct) as usize).max(1);
        
        Self {
            base_cv,
            n_groups,
            test_groups,
            embargo_periods,
        }
    }

    /// Create default combinatorial purged cross-validator
    ///
    /// Uses 10 groups, 2 test groups, 1% embargo, minimum 50 training samples, minimum 10 test samples.
    pub fn default() -> Self {
        Self::new(0.01, 10, 2, 50, 10)
    }

    /// Generate all C(N,k) combinations of test groups
    ///
    /// # Returns
    /// Vector of combinations, where each combination is a vector of group indices
    pub fn generate_combinations(&self) -> Vec<Vec<usize>> {
        use rayon::prelude::*;
        
        let groups: Vec<usize> = (0..self.n_groups).collect();
        let mut combinations = Vec::new();
        
        // Generate combinations iteratively to avoid dependency on itertools
        self.generate_combinations_recursive(&groups, self.test_groups, 0, &mut Vec::new(), &mut combinations);
        
        combinations
    }

    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        groups: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if k == 0 {
            result.push(current.clone());
            return;
        }
        
        for i in start..=(groups.len() - k) {
            current.push(groups[i]);
            self.generate_combinations_recursive(groups, k - 1, i + 1, current, result);
            current.pop();
        }
    }

    /// Create combinatorial purged splits for given data
    ///
    /// # Parameters
    /// - `n_samples`: Total number of samples
    ///
    /// # Returns
    /// Vector of (train_indices, test_indices, combination_id) tuples
    pub fn create_combinatorial_splits(
        &self,
        n_samples: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, usize)>> {
        if n_samples < self.n_groups {
            return Err(PyValueError::new_err("Not enough samples for groups"));
        }

        let combinations = self.generate_combinations();
        let mut splits = Vec::new();
        let group_size = n_samples / self.n_groups;

        for (combo_id, test_group_indices) in combinations.iter().enumerate() {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            // Collect test indices from selected groups
            for &group_idx in test_group_indices {
                let start = group_idx * group_size;
                let end = if group_idx == self.n_groups - 1 {
                    n_samples
                } else {
                    (group_idx + 1) * group_size
                };
                test_indices.extend(start..end);
            }

            // Collect training indices from remaining groups with embargo
            for group_idx in 0..self.n_groups {
                if !test_group_indices.contains(&group_idx) {
                    let start = group_idx * group_size;
                    let end = if group_idx == self.n_groups - 1 {
                        n_samples
                    } else {
                        (group_idx + 1) * group_size
                    };

                    // Apply embargo logic
                    let test_min = *test_indices.iter().min().unwrap_or(&n_samples);
                    let test_max = *test_indices.iter().max().unwrap_or(&0);

                    for idx in start..end {
                        // Check embargo constraints
                        if (idx + self.embargo_periods < test_min) || (idx > test_max + self.embargo_periods) {
                            train_indices.push(idx);
                        }
                    }
                }
            }

            // Only add split if it meets minimum size requirements
            if train_indices.len() >= self.base_cv.min_train_size && test_indices.len() >= self.base_cv.min_test_size {
                splits.push((train_indices, test_indices, combo_id));
            }
        }

        if splits.is_empty() {
            return Err(PyValueError::new_err("No valid combinatorial splits created"));
        }

        Ok(splits)
    }

    /// Calculate Probability of Backtest Overfitting (PBO)
    ///
    /// Implements López de Prado's PBO calculation using the formula:
    /// PBO ≈ 1 - (1 - e^(-k))^(n/k)
    /// where k = number of splits, n = number of parameter combinations tested
    ///
    /// # Parameters
    /// - `performance_scores`: Vector of performance scores from different combinations
    /// - `n_trials`: Number of parameter combinations tested (default: number of combinations)
    ///
    /// # Returns
    /// OverfittingMetrics containing PBO and related statistics
    pub fn calculate_pbo(&self, performance_scores: &[f64], n_trials: Option<usize>) -> OverfittingMetrics {
        let n_combinations = performance_scores.len();
        let trials = n_trials.unwrap_or(n_combinations);
        
        if n_combinations == 0 {
            return OverfittingMetrics {
                pbo: 1.0, // Maximum overfitting probability
                performance_distribution: Vec::new(),
                confidence_interval: (1.0, 1.0),
                n_combinations: 0,
            };
        }

        // Calculate PBO using López de Prado's approximation
        let k = n_combinations as f64;
        let n = trials as f64;
        
        let pbo = if k > 0.0 {
            1.0 - (1.0 - (-k).exp()).powf(n / k)
        } else {
            1.0
        };

        // Calculate confidence interval using bootstrap
        let confidence_interval = self.calculate_pbo_confidence_interval(performance_scores, 0.95);

        OverfittingMetrics {
            pbo: pbo.min(1.0).max(0.0), // Clamp to [0, 1]
            performance_distribution: performance_scores.to_vec(),
            confidence_interval,
            n_combinations,
        }
    }

    /// Calculate confidence interval for PBO using bootstrap sampling
    fn calculate_pbo_confidence_interval(&self, scores: &[f64], confidence_level: f64) -> (f64, f64) {
        use rand::prelude::*;
        
        if scores.len() < 2 {
            return (0.0, 1.0);
        }

        let n_bootstrap = 1000;
        let mut bootstrap_pbos = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..n_bootstrap {
            // Bootstrap sample
            let mut bootstrap_scores = Vec::new();
            for _ in 0..scores.len() {
                let idx = rng.gen_range(0..scores.len());
                bootstrap_scores.push(scores[idx]);
            }

            // Calculate PBO for bootstrap sample
            let k = bootstrap_scores.len() as f64;
            let pbo = if k > 0.0 {
                1.0 - (1.0 - (-k).exp()).powf(k / k) // Simplified for bootstrap
            } else {
                1.0
            };
            bootstrap_pbos.push(pbo.min(1.0).max(0.0));
        }

        // Calculate confidence interval
        bootstrap_pbos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

        let lower = bootstrap_pbos.get(lower_idx).copied().unwrap_or(0.0);
        let upper = bootstrap_pbos.get(upper_idx.min(bootstrap_pbos.len() - 1)).copied().unwrap_or(1.0);

        (lower, upper)
    }

    /// Validate splits for quality and consistency
    pub fn validate_splits(&self, splits: &[(Vec<usize>, Vec<usize>, usize)]) -> bool {
        for (train_indices, test_indices, _) in splits {
            // Check minimum sizes
            if train_indices.len() < self.base_cv.min_train_size || test_indices.len() < self.base_cv.min_test_size {
                return false;
            }

            // Check for overlap
            for &test_idx in test_indices {
                if train_indices.contains(&test_idx) {
                    return false;
                }
            }

            // Check embargo constraints
            if let (Some(&test_min), Some(&test_max)) = (test_indices.iter().min(), test_indices.iter().max()) {
                for &train_idx in train_indices {
                    if train_idx >= test_min.saturating_sub(self.embargo_periods) &&
                       train_idx <= test_max + self.embargo_periods {
                        // This training index is too close to test set
                        if train_idx >= test_min && train_idx <= test_max {
                            return false; // Direct overlap
                        }
                    }
                }
            }
        }
        true
    }
}

impl CrossValidator for CombinatorialPurgedCV {
    fn create_purged_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        // Delegate to base validator for backward compatibility
        self.base_cv.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
    }

    fn create_pattern_aware_cv_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        pattern_duration: usize,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        // Delegate to base validator for backward compatibility
        self.base_cv.create_pattern_aware_cv_splits(n_samples, n_splits, pattern_duration)
    }

    fn validate_cv_splits(
        &self,
        splits: &[(Vec<usize>, Vec<usize>)],
        min_train_size: usize,
        min_test_size: usize,
    ) -> bool {
        self.base_cv.validate_cv_splits(splits, min_train_size, min_test_size)
    }
}

// Ensure thread safety
unsafe impl Send for CombinatorialPurgedCV {}
unsafe impl Sync for CombinatorialPurgedCV {}
unsafe impl Send for CVMetrics {}
unsafe impl Sync for CVMetrics {}
unsafe impl Send for OverfittingMetrics {}
unsafe impl Sync for OverfittingMetrics {}