//! Shared ML components for eliminating code duplication
//!
//! This module provides reusable ML components that can be shared across different
//! model implementations, following the patterns established in the rust_indicators
//! codebase with proper safety using extract_safe! macro and comprehensive error handling.
//!
//! ## Advanced Cross-Validation Integration
//!
//! Advanced Cross-Validation introduces advanced overfitting detection capabilities through:
//! - [`CombinatorialPurgedCV`]: Enhanced cross-validation with C(N,k) combinations
//! - [`OverfittingDetection`]: Statistical overfitting analysis with PBO calculation
//! - Convenience functions for easy integration with existing models
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_indicators::ml::components::{create_advanced_cross_validation_validator, AdvancedCrossValidationConfig};
//!
//! // Create Advanced Cross-Validation validator with default settings
//! let validator = create_advanced_cross_validation_validator(AdvancedCrossValidationConfig::default())?;
//!
//! // Or customize the configuration
//! let config = AdvancedCrossValidationConfig::new()
//!     .embargo_pct(0.02)
//!     .n_groups(10)
//!     .test_groups(2)
//!     .significance_level(0.05);
//! let validator = create_advanced_cross_validation_validator(config)?;
//! ```

use crate::ml::traits::CrossValidator;
use pyo3::prelude::*;

pub mod advanced_cross_validation_integration;
pub mod cross_validation;
pub mod error_handling;
pub mod label_generation;
pub mod memory_optimization;
pub mod overfitting_detection;
pub mod prediction;
pub mod sample_weighting;
pub mod validation_utils;

pub use cross_validation::{
    CVMetrics, CVSplitsOutput, CombinatorialPurgedCV, CombinatorialSplitsOutput,
    OverfittingMetrics, PatternAwareCrossValidator, PurgedCrossValidator,
};
pub use error_handling::{
    ErrorHandler, MLError, MLResult, PerformanceTracker, PredictionResult, ResultBuilder,
    TrainingResults,
};
pub use label_generation::{
    ComponentLabelGenerator, PatternLabeler, TradingSide, TripleBarrierLabeler,
};
pub use memory_optimization::{
    global_memory_optimizer, ArrayOperations, CVMemoryManager, ComputationCache, FeatureProcessor,
    MemoryOptimizer, MemoryPool, MemoryStats, PooledObject,
};
pub use overfitting_detection::{
    DegradationAnalysis, DegradationSeverity, OverfittingDetection, OverfittingReport, PBOResult,
    PerformanceStats,
};
pub use prediction::{BatchPredictor, ConfidencePredictor, MetaLabeler, PredictionEngine};
pub use sample_weighting::{PatternWeighting, SampleWeightCalculator, VolatilityWeighting};
pub use validation_utils::{ModelEvaluators, TrainingWorkflows, ValidationUtils};

/// Configuration builder for Advanced Cross-Validation overfitting detection
#[derive(Debug, Clone)]
pub struct AdvancedCrossValidationConfig {
    pub embargo_pct: f32,
    pub n_groups: usize,
    pub test_groups: usize,
    pub min_train_size: usize,
    pub min_test_size: usize,
    pub significance_level: f64,
    pub n_bootstrap: usize,
}

/// Type alias for the return type of setup_advanced_cross_validation_for_model
pub type AdvancedCrossValidationSetupOutput = (
    CombinatorialPurgedCV,
    OverfittingDetection,
    CombinatorialSplitsOutput,
);

impl Default for AdvancedCrossValidationConfig {
    fn default() -> Self {
        Self {
            embargo_pct: 0.02,
            n_groups: 8,
            test_groups: 2,
            min_train_size: 100,
            min_test_size: 20,
            significance_level: 0.05,
            n_bootstrap: 8,
        }
    }
}

impl AdvancedCrossValidationConfig {
    /// Create a new Advanced Cross-Validation configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embargo percentage (default: 0.02)
    pub fn embargo_pct(mut self, embargo_pct: f32) -> Self {
        self.embargo_pct = embargo_pct;
        self
    }

    /// Set the number of groups for combinatorial splits (default: 8)
    pub fn n_groups(mut self, n_groups: usize) -> Self {
        self.n_groups = n_groups;
        self
    }

    /// Set the number of test groups (default: 2)
    pub fn test_groups(mut self, test_groups: usize) -> Self {
        self.test_groups = test_groups;
        self
    }

    /// Set the minimum training set size (default: 100)
    pub fn min_train_size(mut self, min_train_size: usize) -> Self {
        self.min_train_size = min_train_size;
        self
    }

    /// Set the minimum test set size (default: 20)
    pub fn min_test_size(mut self, min_test_size: usize) -> Self {
        self.min_test_size = min_test_size;
        self
    }

    /// Set the statistical significance level (default: 0.05)
    pub fn significance_level(mut self, significance_level: f64) -> Self {
        self.significance_level = significance_level;
        self
    }

    /// Set the number of bootstrap samples (default: 8)
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.n_bootstrap = n_bootstrap;
        self
    }
}

/// Advanced Cross-Validation validator combining CombinatorialPurgedCV and OverfittingDetection
pub struct AdvancedCrossValidationValidator {
    pub combinatorial_cv: CombinatorialPurgedCV,
    pub overfitting_detector: OverfittingDetection,
}

impl AdvancedCrossValidationValidator {
    /// Create a new Advanced Cross-Validation validator with the given configuration
    pub fn new(config: AdvancedCrossValidationConfig) -> Self {
        Self {
            combinatorial_cv: CombinatorialPurgedCV::new(
                config.embargo_pct,
                config.n_groups,
                config.test_groups,
                config.min_train_size,
                config.min_test_size,
            ),
            overfitting_detector: OverfittingDetection::new(
                config.significance_level,
                config.n_bootstrap,
            ),
        }
    }

    /// Create combinatorial splits for the given sample size
    pub fn create_splits(&self, n_samples: usize) -> PyResult<CombinatorialSplitsOutput> {
        self.combinatorial_cv.create_combinatorial_splits(n_samples)
    }

    /// Calculate PBO (Probability of Backtest Overfitting) for performance data
    pub fn calculate_pbo(&self, in_sample: &[f64], out_sample: &[f64]) -> PyResult<PBOResult> {
        self.overfitting_detector
            .calculate_pbo(in_sample, out_sample)
    }

    /// Get the number of combinations that will be generated
    pub fn get_n_combinations(&self) -> usize {
        // Calculate C(n_groups, test_groups) combinations
        let n = self.combinatorial_cv.n_groups;
        let k = self.combinatorial_cv.test_groups;
        if k > n {
            return 0;
        }

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Check if the configuration is valid for the given sample size
    pub fn validate_config(&self, n_samples: usize) -> bool {
        n_samples
            >= self.combinatorial_cv.base_cv.min_train_size
                + self.combinatorial_cv.base_cv.min_test_size
    }
}

/// Convenience function to create a Advanced Cross-Validation validator with default configuration
pub fn create_advanced_cross_validation_validator(
    config: AdvancedCrossValidationConfig,
) -> PyResult<AdvancedCrossValidationValidator> {
    Ok(AdvancedCrossValidationValidator::new(config))
}

/// Convenience function to create a Advanced Cross-Validation validator with default settings
pub fn create_default_advanced_cross_validation_validator(
) -> PyResult<AdvancedCrossValidationValidator> {
    Ok(AdvancedCrossValidationValidator::new(
        AdvancedCrossValidationConfig::default(),
    ))
}

/// Migration helper: Convert PurgedCrossValidator to CombinatorialPurgedCV
pub fn migrate_to_combinatorial_cv(
    _purged_cv: &PurgedCrossValidator,
    n_groups: usize,
    test_groups: usize,
) -> CombinatorialPurgedCV {
    CombinatorialPurgedCV::new(
        0.02, // Default embargo percentage
        n_groups,
        test_groups,
        100, // Default min train size
        20,  // Default min test size
    )
}

/// Helper function to create a complete Advanced Cross-Validation setup for existing models
pub fn setup_advanced_cross_validation_for_model(
    n_samples: usize,
    config: Option<AdvancedCrossValidationConfig>,
) -> PyResult<AdvancedCrossValidationSetupOutput> {
    let config = config.unwrap_or_default();

    let combinatorial_cv = CombinatorialPurgedCV::new(
        config.embargo_pct,
        config.n_groups,
        config.test_groups,
        config.min_train_size,
        config.min_test_size,
    );

    let overfitting_detector =
        OverfittingDetection::new(config.significance_level, config.n_bootstrap);

    let splits = combinatorial_cv.create_combinatorial_splits(n_samples)?;

    Ok((combinatorial_cv, overfitting_detector, splits))
}

/// Backward compatibility helper: Create traditional purged CV splits
pub fn create_legacy_cv_splits(
    n_samples: usize,
    n_splits: usize,
    embargo_pct: f32,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let purged_cv = PurgedCrossValidator::default();
    purged_cv.create_purged_cv_splits(n_samples, n_splits, embargo_pct)
}

/// Performance comparison helper: Compare Phase 3 vs Advanced Cross-Validation validation
pub struct ValidationComparison {
    pub scientific_labeling_methods_splits: Vec<(Vec<usize>, Vec<usize>)>,
    pub advanced_cross_validation_splits: Vec<(Vec<usize>, Vec<usize>, usize)>,
    pub scientific_labeling_methods_n_splits: usize,
    pub advanced_cross_validation_n_combinations: usize,
}

impl ValidationComparison {
    pub fn new(n_samples: usize, config: AdvancedCrossValidationConfig) -> PyResult<Self> {
        let purged_cv = PurgedCrossValidator::default();
        let scientific_labeling_methods_splits =
            purged_cv.create_purged_cv_splits(n_samples, 5, config.embargo_pct)?;

        let combinatorial_cv = CombinatorialPurgedCV::new(
            config.embargo_pct,
            config.n_groups,
            config.test_groups,
            config.min_train_size,
            config.min_test_size,
        );
        let advanced_cross_validation_splits =
            combinatorial_cv.create_combinatorial_splits(n_samples)?;

        Ok(Self {
            scientific_labeling_methods_n_splits: scientific_labeling_methods_splits.len(),
            advanced_cross_validation_n_combinations: advanced_cross_validation_splits.len(),
            scientific_labeling_methods_splits,
            advanced_cross_validation_splits,
        })
    }

    pub fn get_validation_summary(&self) -> String {
        format!(
            "Scientific Labeling Methods: {} splits | Advanced Cross-Validation: {} combinations ({}x more validation)",
            self.scientific_labeling_methods_n_splits,
            self.advanced_cross_validation_n_combinations,
            self.advanced_cross_validation_n_combinations as f32 / self.scientific_labeling_methods_n_splits as f32
        )
    }
}
