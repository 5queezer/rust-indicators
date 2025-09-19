//! Shared ML components for eliminating code duplication
//!
//! This module provides reusable ML components that can be shared across different
//! model implementations, following the patterns established in the rust_indicators
//! codebase with proper safety using extract_safe! macro and comprehensive error handling.
//!
//! ## Phase 4 Integration
//!
//! Phase 4 introduces advanced overfitting detection capabilities through:
//! - [`CombinatorialPurgedCV`]: Enhanced cross-validation with C(N,k) combinations
//! - [`OverfittingDetection`]: Statistical overfitting analysis with PBO calculation
//! - Convenience functions for easy integration with existing models
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_indicators::ml::components::{create_phase4_validator, Phase4Config};
//!
//! // Create Phase 4 validator with default settings
//! let validator = create_phase4_validator(Phase4Config::default())?;
//!
//! // Or customize the configuration
//! let config = Phase4Config::new()
//!     .embargo_pct(0.02)
//!     .n_groups(10)
//!     .test_groups(2)
//!     .significance_level(0.05);
//! let validator = create_phase4_validator(config)?;
//! ```

use pyo3::prelude::*;
use crate::ml::traits::CrossValidator;

pub mod cross_validation;
pub mod sample_weighting;
pub mod label_generation;
pub mod prediction;
pub mod overfitting_detection;
pub mod phase4_integration;
pub mod validation_utils;
pub mod error_handling;
pub mod memory_optimization;

pub use cross_validation::{
    PurgedCrossValidator,
    PatternAwareCrossValidator,
    CombinatorialPurgedCV,
    CVMetrics,
    OverfittingMetrics
};
pub use sample_weighting::{VolatilityWeighting, PatternWeighting, SampleWeightCalculator};
pub use label_generation::{TradingSide, TripleBarrierLabeler, PatternLabeler, ComponentLabelGenerator};
pub use prediction::{ConfidencePredictor, BatchPredictor, PredictionEngine, MetaLabeler};
pub use overfitting_detection::{
    OverfittingDetection,
    PBOResult,
    OverfittingReport,
    PerformanceStats,
    DegradationAnalysis,
    DegradationSeverity
};
pub use validation_utils::{
    ValidationUtils, ModelEvaluators, TrainingWorkflows,
};
pub use error_handling::{
    MLError, MLResult, TrainingResults, PredictionResult,
    ErrorHandler, PerformanceTracker, ResultBuilder,
};
pub use memory_optimization::{
    MemoryPool, PooledObject, ComputationCache, ArrayOperations,
    FeatureProcessor, CVMemoryManager, MemoryOptimizer, MemoryStats,
    global_memory_optimizer,
};

/// Configuration builder for Phase 4 overfitting detection
#[derive(Debug, Clone)]
pub struct Phase4Config {
    pub embargo_pct: f32,
    pub n_groups: usize,
    pub test_groups: usize,
    pub min_train_size: usize,
    pub min_test_size: usize,
    pub significance_level: f64,
    pub n_bootstrap: usize,
}

impl Default for Phase4Config {
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

impl Phase4Config {
    /// Create a new Phase 4 configuration with default values
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

/// Phase 4 validator combining CombinatorialPurgedCV and OverfittingDetection
pub struct Phase4Validator {
    pub combinatorial_cv: CombinatorialPurgedCV,
    pub overfitting_detector: OverfittingDetection,
}

impl Phase4Validator {
    /// Create a new Phase 4 validator with the given configuration
    pub fn new(config: Phase4Config) -> Self {
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
    pub fn create_splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>, usize)>> {
        self.combinatorial_cv.create_combinatorial_splits(n_samples)
    }

    /// Calculate PBO (Probability of Backtest Overfitting) for performance data
    pub fn calculate_pbo(&self, in_sample: &[f64], out_sample: &[f64]) -> PyResult<PBOResult> {
        self.overfitting_detector.calculate_pbo(in_sample, out_sample)
    }

    /// Get the number of combinations that will be generated
    pub fn get_n_combinations(&self) -> usize {
        // Calculate C(n_groups, test_groups) combinations
        let n = self.combinatorial_cv.n_groups;
        let k = self.combinatorial_cv.test_groups;
        if k > n { return 0; }
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Check if the configuration is valid for the given sample size
    pub fn validate_config(&self, n_samples: usize) -> bool {
        n_samples >= self.combinatorial_cv.base_cv.min_train_size + self.combinatorial_cv.base_cv.min_test_size
    }
}

/// Convenience function to create a Phase 4 validator with default configuration
pub fn create_phase4_validator(config: Phase4Config) -> PyResult<Phase4Validator> {
    Ok(Phase4Validator::new(config))
}

/// Convenience function to create a Phase 4 validator with default settings
pub fn create_default_phase4_validator() -> PyResult<Phase4Validator> {
    Ok(Phase4Validator::new(Phase4Config::default()))
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

/// Helper function to create a complete Phase 4 setup for existing models
pub fn setup_phase4_for_model(
    n_samples: usize,
    config: Option<Phase4Config>,
) -> PyResult<(CombinatorialPurgedCV, OverfittingDetection, Vec<(Vec<usize>, Vec<usize>, usize)>)> {
    let config = config.unwrap_or_default();
    
    let combinatorial_cv = CombinatorialPurgedCV::new(
        config.embargo_pct,
        config.n_groups,
        config.test_groups,
        config.min_train_size,
        config.min_test_size,
    );
    
    let overfitting_detector = OverfittingDetection::new(
        config.significance_level,
        config.n_bootstrap,
    );
    
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

/// Performance comparison helper: Compare Phase 3 vs Phase 4 validation
pub struct ValidationComparison {
    pub phase3_splits: Vec<(Vec<usize>, Vec<usize>)>,
    pub phase4_splits: Vec<(Vec<usize>, Vec<usize>, usize)>,
    pub phase3_n_splits: usize,
    pub phase4_n_combinations: usize,
}

impl ValidationComparison {
    pub fn new(n_samples: usize, config: Phase4Config) -> PyResult<Self> {
        let purged_cv = PurgedCrossValidator::default();
        let phase3_splits = purged_cv.create_purged_cv_splits(n_samples, 5, config.embargo_pct)?;
        
        let combinatorial_cv = CombinatorialPurgedCV::new(
            config.embargo_pct,
            config.n_groups,
            config.test_groups,
            config.min_train_size,
            config.min_test_size,
        );
        let phase4_splits = combinatorial_cv.create_combinatorial_splits(n_samples)?;
        
        Ok(Self {
            phase3_n_splits: phase3_splits.len(),
            phase4_n_combinations: phase4_splits.len(),
            phase3_splits,
            phase4_splits,
        })
    }

    pub fn get_validation_summary(&self) -> String {
        format!(
            "Phase 3: {} splits | Phase 4: {} combinations ({}x more validation)",
            self.phase3_n_splits,
            self.phase4_n_combinations,
            self.phase4_n_combinations as f32 / self.phase3_n_splits as f32
        )
    }
}