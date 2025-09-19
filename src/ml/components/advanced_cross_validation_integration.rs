//! # Advanced Cross-Validation Integration Utilities
//!
//! This module provides shared utilities and traits for integrating Advanced Cross-Validation overfitting
//! detection capabilities across all ML models. It eliminates code duplication and provides
//! a consistent interface for Advanced Cross-Validation functionality.
//!
//! ## Key Components
//!
//! - [`Phase4Capable`]: Trait for models that support Advanced Cross-Validation validation
//! - [`Phase4Config`]: Unified configuration for Advanced Cross-Validation components
//! - [`Phase4Workflow`]: Complete workflow implementation
//! - [`Phase4Results`]: Standardized results structure
//! - [`Phase4Migration`]: Migration utilities for existing models

use crate::ml::components::{CombinatorialPurgedCV, OverfittingDetection, PBOResult};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Unified configuration for Advanced Cross-Validation components
#[derive(Debug, Clone)]
pub struct Phase4Config {
    /// Embargo percentage for purged cross-validation
    pub embargo_pct: f32,
    /// Number of groups for combinatorial splits
    pub n_groups: usize,
    /// Number of test groups in each combination
    pub test_groups: usize,
    /// Minimum training set size
    pub min_train_size: usize,
    /// Minimum test set size
    pub min_test_size: usize,
    /// Statistical significance level
    pub significance_level: f64,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Memory optimization level (0=none, 1=basic, 2=aggressive)
    pub memory_optimization: u8,
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
            parallel_processing: true,
            memory_optimization: 1,
        }
    }
}

impl Phase4Config {
    /// Create a new Advanced Cross-Validation configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern methods for configuration
    pub fn embargo_pct(mut self, embargo_pct: f32) -> Self {
        self.embargo_pct = embargo_pct.clamp(0.0, 0.5);
        self
    }

    pub fn n_groups(mut self, n_groups: usize) -> Self {
        self.n_groups = n_groups.max(3);
        self
    }

    /// Create a builder for Phase4Config
    pub fn builder() -> Phase4ConfigBuilder {
        Phase4ConfigBuilder::new()
    }
    /// Validate configuration for given sample size
    pub fn validate(&self, n_samples: usize) -> Result<(), String> {
        if self.test_groups >= self.n_groups {
            return Err("test_groups must be less than n_groups".to_string());
        }

        let min_required = self.min_train_size + self.min_test_size;
        if n_samples < min_required {
            return Err(format!(
                "Insufficient samples: need at least {}, got {}",
                min_required, n_samples
            ));
        }

        Ok(())
    }

    /// Calculate expected number of combinations
    pub fn calculate_n_combinations(&self) -> usize {
        if self.test_groups > self.n_groups {
            return 0;
        }

        let mut result = 1;
        for i in 0..self.test_groups {
            result = result * (self.n_groups - i) / (i + 1);
        }
        result
    }
}

/// Builder for Phase4Config
pub struct Phase4ConfigBuilder {
    embargo_pct: f32,
    n_groups: usize,
    test_groups: usize,
    min_train_size: usize,
    min_test_size: usize,
    significance_level: f64,
    n_bootstrap: usize,
    memory_optimization: u8,
    parallel_processing: bool,
}

impl Phase4ConfigBuilder {
    pub fn new() -> Self {
        Self {
            embargo_pct: 0.02,
            n_groups: 8,
            test_groups: 2,
            min_train_size: 100,
            min_test_size: 20,
            significance_level: 0.05,
            n_bootstrap: 1000,
            memory_optimization: 1,
            parallel_processing: true,
        }
    }

    pub fn embargo_pct(mut self, embargo_pct: f32) -> Self {
        self.embargo_pct = embargo_pct;
        self
    }

    pub fn n_groups(mut self, n_groups: usize) -> Self {
        self.n_groups = n_groups;
        self
    }

    pub fn test_groups(mut self, test_groups: usize) -> Self {
        self.test_groups = test_groups;
        self
    }

    pub fn min_train_size(mut self, min_train_size: usize) -> Self {
        self.min_train_size = min_train_size;
        self
    }

    pub fn min_test_size(mut self, min_test_size: usize) -> Self {
        self.min_test_size = min_test_size;
        self
    }

    pub fn build(self) -> Phase4Config {
        Phase4Config {
            embargo_pct: self.embargo_pct,
            n_groups: self.n_groups,
            test_groups: self.test_groups,
            min_train_size: self.min_train_size,
            min_test_size: self.min_test_size,
            significance_level: self.significance_level,
            n_bootstrap: self.n_bootstrap,
            parallel_processing: self.parallel_processing,
            memory_optimization: self.memory_optimization,
        }
    }

    pub fn significance_level(mut self, significance_level: f64) -> Self {
        self.significance_level = significance_level.clamp(0.001, 0.1);
        self
    }

    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.n_bootstrap = n_bootstrap.max(3);
        self
    }

    pub fn parallel_processing(mut self, enabled: bool) -> Self {
        self.parallel_processing = enabled;
        self
    }

    pub fn memory_optimization(mut self, level: u8) -> Self {
        self.memory_optimization = level.min(2);
        self
    }
}

/// Standardized Advanced Cross-Validation results structure
#[derive(Debug, Clone)]
pub struct Phase4Results {
    /// Cross-validation performance metrics
    pub cv_metrics: HashMap<String, f32>,
    /// Mean cross-validation score
    pub cv_mean: f32,
    /// Standard deviation of cross-validation scores
    pub cv_std: f32,
    /// Number of splits/combinations processed
    pub n_splits: usize,
    /// PBO analysis results
    pub pbo_result: Option<PBOResult>,
    /// Number of combinations processed
    pub n_combinations: usize,
    /// Training time in milliseconds
    pub training_time_ms: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Validation warnings and recommendations
    pub warnings: Vec<String>,
    /// Performance degradation analysis
    pub degradation_analysis: Option<DegradationAnalysis>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Average memory usage in bytes
    pub avg_memory_bytes: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Memory efficiency score (0-1)
    pub efficiency_score: f32,
}

/// Performance degradation analysis
#[derive(Debug, Clone)]
pub struct DegradationAnalysis {
    /// Mean degradation from in-sample to out-of-sample
    pub mean_degradation: f64,
    /// Standard deviation of degradation
    pub std_degradation: f64,
    /// Percentage of combinations showing degradation
    pub degradation_frequency: f64,
    /// Severity classification
    pub severity: DegradationSeverity,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Classification of degradation severity
#[derive(Debug, Clone, PartialEq)]
pub enum DegradationSeverity {
    Low,
    Moderate,
    High,
    Severe,
}

/// Trait for models that support Advanced Cross-Validation validation
pub trait Phase4Capable {
    /// Enable Advanced Cross-Validation validation with given configuration
    fn enable_advanced_cross_validation(&mut self, config: Phase4Config) -> PyResult<()>;

    /// Check if Advanced Cross-Validation is enabled
    fn is_advanced_cross_validation_enabled(&self) -> bool;

    /// Get current Advanced Cross-Validation configuration
    fn get_advanced_cross_validation_config(&self) -> Option<&Phase4Config>;

    /// Train with Advanced Cross-Validation enhanced validation
    fn train_with_advanced_cross_validation(
        &mut self,
        features: &pyo3::Bound<'_, PyArray2<f32>>,
        labels: &pyo3::Bound<'_, PyArray1<i32>>,
        learning_rate: f32,
    ) -> PyResult<Phase4Results>;

    /// Get comprehensive overfitting analysis
    fn get_overfitting_analysis(&self) -> PyResult<Option<HashMap<String, f32>>>;

    /// Assess overfitting risk with human-readable output
    fn assess_overfitting_risk(&self) -> String;
}

/// Complete Advanced Cross-Validation workflow implementation
#[derive(Clone)]
pub struct Phase4Workflow {
    pub config: Phase4Config,
    pub combinatorial_cv: CombinatorialPurgedCV,
    pub overfitting_detector: OverfittingDetection,
    memory_tracker: MemoryTracker,
}

impl Phase4Workflow {
    /// Create a new Advanced Cross-Validation workflow with given configuration
    pub fn new(config: Phase4Config) -> Self {
        let combinatorial_cv = CombinatorialPurgedCV::new(
            config.embargo_pct,
            config.n_groups,
            config.test_groups,
            config.min_train_size,
            config.min_test_size,
        );

        let overfitting_detector =
            OverfittingDetection::new(config.significance_level, config.n_bootstrap);

        let memory_tracker = MemoryTracker::new(config.memory_optimization);

        Self {
            config,
            combinatorial_cv,
            overfitting_detector,
            memory_tracker,
        }
    }

    /// Execute complete Advanced Cross-Validation validation workflow
    pub fn execute_validation<F>(
        &mut self,
        n_samples: usize,
        mut train_eval_fn: F,
    ) -> PyResult<Phase4Results>
    where
        F: FnMut(&[usize], &[usize], usize) -> PyResult<(f32, f32)>, // (train_score, test_score)
    {
        let start_time = std::time::Instant::now();
        self.memory_tracker.start_tracking();

        // Validate configuration
        self.config
            .validate(n_samples)
            .map_err(|e| PyValueError::new_err(e))?;

        // Generate combinatorial splits
        let splits = self
            .combinatorial_cv
            .create_combinatorial_splits(n_samples)?;
        let n_combinations = splits.len();

        // Execute validation across all combinations
        let mut cv_scores = Vec::new();
        let mut train_scores = Vec::new();
        let mut warnings = Vec::new();

        for (train_idx, test_idx, combo_id) in &splits {
            match train_eval_fn(train_idx, test_idx, *combo_id) {
                Ok((train_score, test_score)) => {
                    cv_scores.push(test_score as f64);
                    train_scores.push(train_score as f64);

                    // Track memory usage
                    self.memory_tracker.record_iteration();
                }
                Err(e) => {
                    warnings.push(format!("Combination {} failed: {}", combo_id, e));
                }
            }
        }

        if cv_scores.is_empty() {
            return Err(PyValueError::new_err("No valid combinations completed"));
        }

        // Calculate performance metrics
        let cv_mean = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
        let cv_std = {
            let variance = cv_scores.iter().map(|x| (x - cv_mean).powi(2)).sum::<f64>()
                / cv_scores.len() as f64;
            variance.sqrt()
        };

        let mut cv_metrics = HashMap::new();
        cv_metrics.insert("cv_mean".to_string(), cv_mean as f32);
        cv_metrics.insert("cv_std".to_string(), cv_std as f32);
        cv_metrics.insert("n_combinations".to_string(), n_combinations as f32);

        // Calculate PBO if we have sufficient data
        let pbo_result = if cv_scores.len() >= self.config.n_bootstrap {
            match self
                .overfitting_detector
                .calculate_pbo(&train_scores, &cv_scores)
            {
                Ok(pbo) => {
                    cv_metrics.insert("pbo_value".to_string(), pbo.pbo_value as f32);
                    cv_metrics.insert(
                        "is_overfit".to_string(),
                        if pbo.is_overfit { 1.0 } else { 0.0 },
                    );
                    Some(pbo)
                }
                Err(e) => {
                    warnings.push(format!("PBO calculation failed: {}", e));
                    None
                }
            }
        } else {
            warnings.push("Insufficient combinations for reliable PBO calculation".to_string());
            None
        };

        // Analyze performance degradation
        let degradation_analysis = self.analyze_degradation(&train_scores, &cv_scores);

        // Finalize memory tracking
        let memory_stats = self.memory_tracker.finalize();
        let training_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(Phase4Results {
            cv_metrics,
            cv_mean: cv_mean as f32,
            cv_std: cv_std as f32,
            n_splits: n_combinations,
            pbo_result,
            n_combinations,
            training_time_ms,
            memory_stats,
            warnings,
            degradation_analysis: Some(degradation_analysis),
        })
    }

    /// Analyze performance degradation patterns
    fn analyze_degradation(
        &self,
        train_scores: &[f64],
        test_scores: &[f64],
    ) -> DegradationAnalysis {
        let degradations: Vec<f64> = train_scores
            .iter()
            .zip(test_scores.iter())
            .map(|(train, test)| train - test)
            .collect();

        let mean_degradation = degradations.iter().sum::<f64>() / degradations.len() as f64;
        let variance = degradations
            .iter()
            .map(|d| (d - mean_degradation).powi(2))
            .sum::<f64>()
            / degradations.len() as f64;
        let std_degradation = variance.sqrt();

        let degradation_count = degradations.iter().filter(|&&d| d > 0.0).count();
        let degradation_frequency = degradation_count as f64 / degradations.len() as f64;

        // Classify severity
        let severity = match mean_degradation {
            x if x > 0.2 => DegradationSeverity::Severe,
            x if x > 0.1 => DegradationSeverity::High,
            x if x > 0.05 => DegradationSeverity::Moderate,
            _ => DegradationSeverity::Low,
        };

        // Generate recommendations
        let mut recommendations = Vec::new();
        match severity {
            DegradationSeverity::Severe => {
                recommendations.push(
                    "Critical overfitting detected. Reduce model complexity immediately."
                        .to_string(),
                );
                recommendations.push("Consider ensemble methods or regularization.".to_string());
            }
            DegradationSeverity::High => {
                recommendations.push(
                    "Significant overfitting. Add regularization or reduce features.".to_string(),
                );
            }
            DegradationSeverity::Moderate => {
                recommendations
                    .push("Moderate overfitting. Monitor performance closely.".to_string());
            }
            DegradationSeverity::Low => {
                recommendations.push("Good generalization. Model appears robust.".to_string());
            }
        }

        if degradation_frequency > 0.8 {
            recommendations
                .push("High degradation frequency suggests systematic overfitting.".to_string());
        }

        DegradationAnalysis {
            mean_degradation,
            std_degradation,
            degradation_frequency,
            severity,
            recommendations,
        }
    }
}

/// Memory tracking utility
#[derive(Clone)]
struct MemoryTracker {
    #[allow(dead_code)]
    optimization_level: u8,
    start_memory: usize,
    peak_memory: usize,
    total_memory: usize,
    iterations: usize,
    allocations: usize,
}

impl MemoryTracker {
    fn new(optimization_level: u8) -> Self {
        Self {
            optimization_level,
            start_memory: 0,
            peak_memory: 0,
            total_memory: 0,
            iterations: 0,
            allocations: 0,
        }
    }

    fn start_tracking(&mut self) {
        // In a real implementation, this would use system memory APIs
        self.start_memory = self.get_current_memory_usage();
    }

    fn record_iteration(&mut self) {
        let current_memory = self.get_current_memory_usage();
        self.peak_memory = self.peak_memory.max(current_memory);
        self.total_memory += current_memory;
        self.iterations += 1;
        self.allocations += 1; // Simplified
    }

    fn finalize(&self) -> MemoryStats {
        let avg_memory = if self.iterations > 0 {
            self.total_memory / self.iterations
        } else {
            0
        };

        let efficiency_score = if self.peak_memory > 0 {
            (avg_memory as f32 / self.peak_memory as f32).min(1.0)
        } else {
            1.0
        };

        MemoryStats {
            peak_memory_bytes: self.peak_memory,
            avg_memory_bytes: avg_memory,
            allocations: self.allocations,
            efficiency_score,
        }
    }

    fn get_current_memory_usage(&self) -> usize {
        // Simplified memory tracking - in production would use actual system APIs
        std::mem::size_of::<Self>() * (self.iterations + 1)
    }
}

/// Migration utilities for existing models
pub struct Phase4Migration;

impl Phase4Migration {
    /// Migrate existing PurgedCrossValidator to CombinatorialPurgedCV
    pub fn migrate_cross_validator(
        embargo_pct: f32,
        min_train_size: usize,
        min_test_size: usize,
    ) -> CombinatorialPurgedCV {
        CombinatorialPurgedCV::new(
            embargo_pct,
            8, // Default n_groups
            2, // Default test_groups
            min_train_size,
            min_test_size,
        )
    }

    /// Create Advanced Cross-Validation configuration from legacy parameters
    pub fn create_config_from_legacy(
        embargo_pct: f32,
        n_splits: usize,
        min_train_size: usize,
        min_test_size: usize,
    ) -> Phase4Config {
        Phase4Config::builder()
            .embargo_pct(embargo_pct)
            .n_groups(n_splits.max(5))
            .test_groups(2)
            .min_train_size(min_train_size)
            .min_test_size(min_test_size)
            .build()
    }

    /// Generate migration report for existing model
    pub fn generate_migration_report(
        current_cv_splits: usize,
        proposed_config: &Phase4Config,
    ) -> String {
        let n_combinations = proposed_config.calculate_n_combinations();
        let improvement_factor = n_combinations as f32 / current_cv_splits as f32;

        format!(
            "Advanced Cross-Validation Migration Report:\n\
            Current CV splits: {}\n\
            Proposed combinations: {}\n\
            Validation improvement: {:.1}x\n\
            Expected PBO accuracy: {}\n\
            Memory overhead: {}\n\
            Recommended: {}",
            current_cv_splits,
            n_combinations,
            improvement_factor,
            if n_combinations >= 10 {
                "High"
            } else {
                "Medium"
            },
            match proposed_config.memory_optimization {
                0 => "None",
                1 => "Basic",
                2 => "Aggressive",
                _ => "Unknown",
            },
            if improvement_factor > 2.0 {
                "Yes - significant improvement"
            } else {
                "Consider for robustness"
            }
        )
    }
}

/// Convenience functions for common Advanced Cross-Validation operations
pub mod convenience {
    use super::*;

    /// Create a Advanced Cross-Validation workflow with sensible defaults for pattern recognition
    pub fn create_pattern_workflow() -> Phase4Workflow {
        let config = Phase4Config::builder()
            .embargo_pct(0.01) // Lower embargo for patterns
            .n_groups(6)
            .test_groups(2)
            .min_train_size(80)
            .min_test_size(15)
            .build();

        Phase4Workflow::new(config)
    }

    /// Create a Advanced Cross-Validation workflow with sensible defaults for trading classification
    pub fn create_trading_workflow() -> Phase4Workflow {
        let config = Phase4Config::builder()
            .embargo_pct(0.02) // Higher embargo for trading
            .n_groups(8)
            .test_groups(2)
            .min_train_size(100)
            .min_test_size(20)
            .build();

        Phase4Workflow::new(config)
    }

    /// Create a Advanced Cross-Validation workflow with sensible defaults for unified models
    pub fn create_unified_workflow() -> Phase4Workflow {
        let config = Phase4Config::builder()
            .embargo_pct(0.015) // Balanced embargo
            .n_groups(10)
            .test_groups(3) // More test groups for comprehensive validation
            .min_train_size(120)
            .min_test_size(25)
            .build();

        Phase4Workflow::new(config)
    }

    /// Quick Advanced Cross-Validation setup for existing models
    pub fn quick_setup(
        n_samples: usize,
        model_type: &str,
    ) -> PyResult<(Phase4Config, CombinatorialPurgedCV, OverfittingDetection)> {
        let config = match model_type {
            "pattern" => create_pattern_workflow().config,
            "trading" => create_trading_workflow().config,
            "unified" => create_unified_workflow().config,
            _ => Phase4Config::default(),
        };

        config
            .validate(n_samples)
            .map_err(|e| PyValueError::new_err(e))?;

        let combinatorial_cv = CombinatorialPurgedCV::new(
            config.embargo_pct,
            config.n_groups,
            config.test_groups,
            config.min_train_size,
            config.min_test_size,
        );

        let overfitting_detector =
            OverfittingDetection::new(config.significance_level, config.n_bootstrap);

        Ok((config, combinatorial_cv, overfitting_detector))
    }
}

// Ensure thread safety
unsafe impl Send for Phase4Config {}
unsafe impl Sync for Phase4Config {}
unsafe impl Send for Phase4Results {}
unsafe impl Sync for Phase4Results {}
unsafe impl Send for Phase4Workflow {}
unsafe impl Sync for Phase4Workflow {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_cross_validation_config_validation() {
        let config = Phase4Config::builder()
            .n_groups(5)
            .test_groups(2)
            .min_train_size(50)
            .min_test_size(10)
            .build();

        assert!(config.validate(100).is_ok());
        assert!(config.validate(50).is_err()); // Too few samples
    }

    #[test]
    fn test_advanced_cross_validation_config_combinations() {
        let config = Phase4Config::builder().n_groups(5).test_groups(2).build();

        assert_eq!(config.calculate_n_combinations(), 10); // C(5,2) = 10
    }

    #[test]
    fn test_migration_utilities() {
        let config = Phase4Migration::create_config_from_legacy(0.02, 5, 100, 20);
        assert_eq!(config.embargo_pct, 0.02);
        assert_eq!(config.n_groups, 5);
        assert_eq!(config.min_train_size, 100);
        assert_eq!(config.min_test_size, 20);
    }
}
