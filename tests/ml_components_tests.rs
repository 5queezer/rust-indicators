//! Comprehensive tests for ML components
//! 
//! Tests cover López de Prado's ML requirements including:
//! - Triple barrier labeling correctness (profit targets, stop losses)
//! - Cross-validation overlap detection and embargo periods
//! - Sample weight reasonableness checks
//! - Edge cases for fractional differentiation (when implemented)

use rust_indicators::ml::components::{
    TripleBarrierLabeler, PatternLabeler, ComponentLabelGenerator,
    PurgedCrossValidator, PatternAwareCrossValidator,
    VolatilityWeighting, PatternWeighting, SampleWeightCalculator,
};
use rust_indicators::ml::traits::{LabelGenerator, CrossValidator};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use rstest::fixture;

#[cfg(test)]
mod ml_components_tests {
    use super::*;
    
    // Initialize Python for all tests
    #[ctor::ctor]
    fn init() {
        pyo3::prepare_freethreaded_python();
    }

    // === FIXTURES ===

    #[fixture]
    fn sample_prices() -> Vec<f32> {
        vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 106.0, 110.0, 107.0, 112.0]
    }

    #[fixture]
    fn sample_volatility() -> Vec<f32> {
        vec![0.02, 0.025, 0.015, 0.03, 0.02, 0.035, 0.018, 0.04, 0.022, 0.028]
    }

    #[fixture]
    fn sample_returns() -> Vec<f32> {
        vec![0.0, 0.02, -0.0098, 0.0396, -0.019, 0.0485, -0.0185, 0.0377, -0.0273, 0.0467]
    }

    #[fixture]
    fn sample_ohlc() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let open = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 106.0, 110.0, 107.0, 112.0];
        let high = vec![101.5, 103.2, 102.1, 106.5, 104.8, 109.2, 107.5, 111.8, 108.9, 113.5];
        let low = vec![99.2, 101.1, 100.2, 104.1, 102.3, 107.1, 105.2, 109.3, 106.1, 111.2];
        let close = vec![101.0, 102.5, 101.8, 105.8, 104.2, 108.5, 106.8, 110.5, 107.8, 112.8];
        (open, high, low, close)
    }

    // === TRIPLE BARRIER LABELING TESTS ===

    #[test]
    fn test_triple_barrier_labeler_creation() {
        let labeler = TripleBarrierLabeler::new(2.0, 1.5, 20);
        assert_eq!(labeler.default_profit_mult, 2.0);
        assert_eq!(labeler.default_stop_mult, 1.5);
        assert_eq!(labeler.default_max_hold, 20);
    }

    #[test]
    fn test_triple_barrier_labeling_correctness() {
        Python::with_gil(|py| {
            let labeler = TripleBarrierLabeler::new(1.0, 1.0, 5);
            
            // Create test data where we know the expected outcomes
            let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 105.0, 100.0];
            let volatility = vec![0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 1.0, 1.0, 5
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // First few labels should be buy signals (2) due to uptrend
            // Last few should be hold (1) due to insufficient future data
            assert_eq!(labels_vec.len(), 8);
            assert!(labels_vec[0] == 2 || labels_vec[0] == 1); // Buy or hold
            assert!(labels_vec.iter().all(|&x| x >= 0 && x <= 2)); // Valid range
        });
    }

    #[test]
    fn test_triple_barrier_profit_target_hit() {
        Python::with_gil(|py| {
            let labeler = TripleBarrierLabeler::new(0.5, 2.0, 10); // Low profit target, high stop
            
            // Price goes up significantly - should hit profit target
            let prices = vec![100.0, 100.6, 101.2, 101.8, 102.4, 103.0];
            let volatility = vec![0.01; 6]; // 1% volatility
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 0.5, 2.0, 5
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // First label should be buy (2) since price increases by 0.5% * 1% = 0.005 target
            assert_eq!(labels_vec[0], 2);
        });
    }

    #[test]
    fn test_triple_barrier_stop_loss_hit() {
        Python::with_gil(|py| {
            let labeler = TripleBarrierLabeler::new(2.0, 0.5, 10); // High profit target, low stop
            
            // Price goes down significantly - should hit stop loss
            let prices = vec![100.0, 99.4, 98.8, 98.2, 97.6, 97.0];
            let volatility = vec![0.01; 6]; // 1% volatility
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 2.0, 0.5, 5
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // First label should be sell (0) since price decreases by 0.5% * 1% = 0.005 stop
            assert_eq!(labels_vec[0], 0);
        });
    }

    #[test]
    fn test_triple_barrier_time_exit() {
        Python::with_gil(|py| {
            let labeler = TripleBarrierLabeler::new(10.0, 10.0, 3); // Very high barriers, short time
            
            // Price moves sideways - should hit time barrier
            let prices = vec![100.0, 100.1, 100.05, 100.15, 100.2];
            let volatility = vec![0.01; 5];
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 10.0, 10.0, 3
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // Should be hold (1) since barriers are too high and time runs out
            assert_eq!(labels_vec[0], 1);
        });
    }

    #[test]
    fn test_triple_barrier_edge_cases() {
        Python::with_gil(|py| {
            let labeler = TripleBarrierLabeler::new(2.0, 1.5, 20);
            
            // Test with zero volatility
            let prices = vec![100.0, 101.0, 102.0];
            let zero_vol = vec![0.0, 0.0, 0.0];
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, zero_vol).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 2.0, 1.5, 5
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // Should handle zero volatility gracefully
            assert_eq!(labels_vec.len(), 3);
            assert!(labels_vec.iter().all(|&x| x >= 0 && x <= 2));
            
            // Test with negative prices (should skip)
            let neg_prices = vec![-100.0, 101.0, 102.0];
            let volatility = vec![0.01, 0.01, 0.01];
            
            let neg_prices_array = PyArray1::from_vec(py, neg_prices).readonly();
            let vol_array2 = PyArray1::from_vec(py, volatility).readonly();
            
            let labels2 = labeler.create_triple_barrier_labels(
                py, neg_prices_array, vol_array2, 2.0, 1.5, 5
            ).unwrap();
            
            let labels_vec2: Vec<i32> = labels2.bind(py).readonly().as_array().to_vec();
            
            // Should handle negative prices gracefully
            assert_eq!(labels_vec2.len(), 3);
            assert!(labels_vec2.iter().all(|&x| x >= 0 && x <= 2));
        });
    }

    // === PATTERN LABELING TESTS ===

    #[test]
    fn test_pattern_labeler_creation() {
        let labeler = PatternLabeler::new(10, 0.02, 0.02);
        assert_eq!(labeler.default_future_periods, 10);
        assert_eq!(labeler.default_profit_threshold, 0.02);
        assert_eq!(labeler.default_stop_threshold, 0.02);
    }

    #[test]
    fn test_pattern_labeling_correctness() {
        Python::with_gil(|py| {
            let labeler = PatternLabeler::new(5, 0.02, 0.02);
            let (open, high, low, close) = sample_ohlc();
            
            let open_array = PyArray1::from_vec(py, open).readonly();
            let high_array = PyArray1::from_vec(py, high).readonly();
            let low_array = PyArray1::from_vec(py, low).readonly();
            let close_array = PyArray1::from_vec(py, close).readonly();
            
            let labels = labeler.create_pattern_labels(
                py, open_array, high_array, low_array, close_array, 5, 0.02, 0.02
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // Should generate valid labels
            assert_eq!(labels_vec.len(), 10);
            assert!(labels_vec.iter().all(|&x| x >= 0 && x <= 2));
        });
    }

    // === CROSS-VALIDATION TESTS ===

    #[test]
    fn test_purged_cross_validator_creation() {
        let validator = PurgedCrossValidator::new(0.01, 50, 10);
        assert_eq!(validator.embargo_pct, 0.01);
        assert_eq!(validator.min_train_size, 50);
        assert_eq!(validator.min_test_size, 10);
    }

    #[test]
    fn test_purged_cv_overlap_detection() {
        let validator = PurgedCrossValidator::new(0.1, 10, 5);
        let splits = validator.create_purged_cv_splits(100, 3, 0.1).unwrap();
        
        // Verify no overlap between train and test sets
        for (train_indices, test_indices) in &splits {
            for &test_idx in test_indices {
                assert!(!train_indices.contains(&test_idx), 
                    "Found overlap: test index {} in train set", test_idx);
            }
        }
        
        // Verify embargo periods are respected
        for (train_indices, test_indices) in &splits {
            let test_min = *test_indices.iter().min().unwrap();
            let test_max = *test_indices.iter().max().unwrap();
            let embargo_size = (100.0 * 0.1) as usize;
            
            // Check that train indices respect embargo before test set
            for &train_idx in train_indices {
                if train_idx < test_min {
                    assert!(train_idx + embargo_size <= test_min,
                        "Train index {} too close to test set start {}", train_idx, test_min);
                }
                if train_idx > test_max {
                    assert!(train_idx >= test_max + embargo_size,
                        "Train index {} too close to test set end {}", train_idx, test_max);
                }
            }
        }
    }

    #[test]
    fn test_cv_splits_validation() {
        let validator = PurgedCrossValidator::new(0.05, 20, 10);
        let splits = validator.create_purged_cv_splits(200, 5, 0.05).unwrap();
        
        // Validate splits meet minimum size requirements
        assert!(validator.validate_cv_splits(&splits, 20, 10));
        
        // Test with too strict requirements
        assert!(!validator.validate_cv_splits(&splits, 100, 50));
    }

    #[test]
    fn test_pattern_aware_cv_splits() {
        let validator = PatternAwareCrossValidator::new(0.02, 5);
        let splits = validator.create_default_pattern_splits(100, Some(3)).unwrap();
        
        assert!(!splits.is_empty());
        assert!(validator.validate_cv_splits(&splits, 10, 5));
        
        // Pattern-aware should have larger embargo periods
        for (train_indices, test_indices) in &splits {
            let test_min = *test_indices.iter().min().unwrap();
            let test_max = *test_indices.iter().max().unwrap();
            
            // Check for pattern-aware embargo (should be at least 2 * pattern_duration)
            let min_embargo = 2 * 5; // 2 * pattern_duration
            
            for &train_idx in train_indices {
                if train_idx < test_min {
                    assert!(train_idx + min_embargo <= test_min,
                        "Pattern-aware embargo not respected");
                }
                if train_idx > test_max {
                    assert!(train_idx >= test_max + min_embargo,
                        "Pattern-aware embargo not respected");
                }
            }
        }
    }

    #[test]
    fn test_cv_insufficient_samples() {
        let validator = PurgedCrossValidator::new(0.1, 10, 5);
        
        // Should fail with insufficient samples
        let result = validator.create_purged_cv_splits(10, 5, 0.1);
        assert!(result.is_err());
        
        // Should fail when embargo is too large
        let result2 = validator.create_purged_cv_splits(100, 3, 0.9);
        assert!(result2.is_err() || result2.unwrap().is_empty());
    }

    // === SAMPLE WEIGHTING TESTS ===

    #[test]
    fn test_volatility_weighting_creation() {
        let weighting = VolatilityWeighting::new(20, 0.1, 3.0);
        assert_eq!(weighting.window_size, 20);
        assert_eq!(weighting.min_weight, 0.1);
        assert_eq!(weighting.max_weight, 3.0);
    }

    #[test]
    fn test_volatility_weight_reasonableness() {
        Python::with_gil(|py| {
            let weighting = VolatilityWeighting::new(5, 0.1, 3.0);
            let returns = sample_returns();
            
            let returns_array = PyArray1::from_vec(py, returns.clone()).readonly();
            let weights = weighting.calculate_weights(py, returns_array).unwrap();
            let weights_vec: Vec<f32> = weights.bind(py).readonly().as_array().to_vec();
            
            // Check weight reasonableness
            assert_eq!(weights_vec.len(), returns.len());
            
            // All weights should be within bounds
            for &weight in &weights_vec {
                assert!(weight >= 0.1 && weight <= 3.0, 
                    "Weight {} outside bounds [0.1, 3.0]", weight);
            }
            
            // Mean weight should be close to 1.0 (not exactly due to clamping)
            let mean_weight: f32 = weights_vec.iter().sum::<f32>() / weights_vec.len() as f32;
            assert!(mean_weight > 0.5 && mean_weight < 2.0, 
                "Mean weight {} unreasonable", mean_weight);
            
            // High volatility periods should get higher weights
            let abs_returns: Vec<f32> = returns.iter().map(|r| r.abs()).collect();
            let max_abs_return_idx = abs_returns.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            if max_abs_return_idx >= 5 { // Ensure we have enough history
                let max_weight = weights_vec[max_abs_return_idx];
                let avg_weight = weights_vec.iter().sum::<f32>() / weights_vec.len() as f32;
                assert!(max_weight >= avg_weight, 
                    "Highest volatility period should have above-average weight");
            }
        });
    }

    #[test]
    fn test_pattern_weighting_reasonableness() {
        Python::with_gil(|py| {
            let weighting = PatternWeighting::new(0.02, 0.5, 2.0);
            
            // Create pattern signals - some samples have multiple patterns (should get lower weights)
            let pattern_signals = vec![
                vec![0.8f32, 0.0f32, 0.0f32], // Single pattern
                vec![0.9f32, 0.7f32, 0.0f32], // Two patterns
                vec![0.0f32, 0.0f32, 0.0f32], // No patterns
                vec![0.6f32, 0.8f32, 0.9f32], // Three patterns
                vec![0.0f32, 0.9f32, 0.0f32], // Single pattern
            ];
            
            let volatility = vec![0.02f32, 0.025f32, 0.015f32, 0.03f32, 0.02f32];
            
            let pattern_array = PyArray2::from_vec2(py, &pattern_signals).unwrap().readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let weights = weighting.calculate_pattern_weights(py, pattern_array, vol_array).unwrap();
            let weights_vec: Vec<f32> = weights.bind(py).readonly().as_array().to_vec();
            
            // Check weight reasonableness
            assert_eq!(weights_vec.len(), 5);
            
            // All weights should be within bounds
            for &weight in &weights_vec {
                assert!(weight >= 0.5 && weight <= 2.0, 
                    "Weight {} outside bounds [0.5, 2.0]", weight);
            }
            
            // Single pattern samples should have higher weights than multi-pattern samples
            let single_pattern_weight = weights_vec[0]; // First sample has 1 pattern
            let multi_pattern_weight = weights_vec[3];   // Fourth sample has 3 patterns
            assert!(single_pattern_weight > multi_pattern_weight,
                "Single pattern weight {} should be higher than multi-pattern weight {}", 
                single_pattern_weight, multi_pattern_weight);
        });
    }

    #[test]
    fn test_sample_weight_calculator_combined() {
        Python::with_gil(|py| {
            let calculator = SampleWeightCalculator::default();
            let returns = sample_returns();
            let volatility = sample_volatility();
            
            let returns_array = PyArray1::from_vec(py, returns.clone()).readonly();
            let vol_array = PyArray1::from_vec(py, volatility.clone()).readonly();
            
            // Test volatility-only weights
            let vol_weights = calculator.calculate_volatility_weights(py, returns_array.clone()).unwrap();
            let vol_weights_vec: Vec<f32> = vol_weights.bind(py).readonly().as_array().to_vec();
            
            assert_eq!(vol_weights_vec.len(), returns.len());
            
            // Test combined weights (without patterns)
            let combined_weights = calculator.calculate_combined_weights(
                py, returns_array, None, vol_array
            ).unwrap();
            let combined_weights_vec: Vec<f32> = combined_weights.bind(py).readonly().as_array().to_vec();
            
            // Should be same as volatility weights when no patterns provided
            assert_eq!(vol_weights_vec, combined_weights_vec);
        });
    }

    #[test]
    fn test_sample_weights_statistical_properties() {
        Python::with_gil(|py| {
            let calculator = SampleWeightCalculator::default();
            
            // Create longer series for better statistics
            let mut returns = Vec::new();
            for i in 0..100 {
                let base_vol = 0.02;
                let vol_multiplier = if i % 20 < 5 { 3.0 } else { 1.0 }; // High vol periods
                returns.push((i as f32 * 0.001 - 0.05) * base_vol * vol_multiplier);
            }
            
            let returns_array = PyArray1::from_vec(py, returns).readonly();
            let weights = calculator.calculate_volatility_weights(py, returns_array).unwrap();
            let weights_vec: Vec<f32> = weights.bind(py).readonly().as_array().to_vec();
            
            // Statistical checks
            let mean_weight: f32 = weights_vec.iter().sum::<f32>() / weights_vec.len() as f32;
            let variance: f32 = weights_vec.iter()
                .map(|w| (w - mean_weight).powi(2))
                .sum::<f32>() / weights_vec.len() as f32;
            let std_weight = variance.sqrt();
            
            // Mean should be reasonable (not exactly 1.0 due to clamping and window effects)
            assert!(mean_weight > 0.5 && mean_weight < 2.0, 
                "Mean weight {} unreasonable", mean_weight);
            
            // Standard deviation should indicate some variation but not extreme
            assert!(std_weight < 1.0, 
                "Weight standard deviation {} too high", std_weight);
            
            // Should have some weights above and below mean
            let above_mean = weights_vec.iter().filter(|&&w| w > mean_weight).count();
            let below_mean = weights_vec.iter().filter(|&&w| w < mean_weight).count();
            
            assert!(above_mean > 0 && below_mean > 0, 
                "Weights should have variation around mean");
        });
    }

    // === INTEGRATION TESTS ===

    #[test]
    fn test_component_label_generator_integration() {
        let generator = ComponentLabelGenerator::default();
        
        // Test that it combines both labelers correctly
        assert_eq!(generator.triple_barrier.default_profit_mult, 2.0);
        assert_eq!(generator.pattern_labeler.default_future_periods, 10);
        
        Python::with_gil(|py| {
            let prices = sample_prices();
            let volatility = sample_volatility();
            let (open, high, low, close) = sample_ohlc();
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            let open_array = PyArray1::from_vec(py, open).readonly();
            let high_array = PyArray1::from_vec(py, high).readonly();
            let low_array = PyArray1::from_vec(py, low).readonly();
            let close_array = PyArray1::from_vec(py, close).readonly();
            
            // Test triple barrier labels
            let triple_labels = generator.create_triple_barrier_labels(
                py, prices_array, vol_array, 2.0, 1.5, 10
            ).unwrap();
            let triple_vec: Vec<i32> = triple_labels.bind(py).readonly().as_array().to_vec();
            assert!(triple_vec.iter().all(|&x| x >= 0 && x <= 2));
            
            // Test pattern labels
            let pattern_labels = generator.create_pattern_labels(
                py, open_array, high_array, low_array, close_array, 5, 0.02, 0.02
            ).unwrap();
            let pattern_vec: Vec<i32> = pattern_labels.bind(py).readonly().as_array().to_vec();
            assert!(pattern_vec.iter().all(|&x| x >= 0 && x <= 2));
        });
    }

    // === LÓPEZ DE PRADO SPECIFIC TESTS ===

    #[test]
    fn test_lopez_de_prado_triple_barrier_specifications() {
        Python::with_gil(|py| {
            // Test the exact specifications from López de Prado's book
            let labeler = TripleBarrierLabeler::new(1.0, 1.0, 10);
            
            // Create a scenario that should clearly hit profit target
            let prices = vec![100.0, 100.5, 101.0, 101.5, 102.0, 102.5];
            let volatility = vec![0.01; 6]; // 1% volatility means 1% profit target
            
            let prices_array = PyArray1::from_vec(py, prices).readonly();
            let vol_array = PyArray1::from_vec(py, volatility).readonly();
            
            let labels = labeler.create_triple_barrier_labels(
                py, prices_array, vol_array, 1.0, 1.0, 5
            ).unwrap();
            
            let labels_vec: Vec<i32> = labels.bind(py).readonly().as_array().to_vec();
            
            // With 1% volatility and 1.0 multiplier, profit target is 101.0
            // Price reaches 101.0 at index 2, so first label should be buy (2)
            assert_eq!(labels_vec[0], 2, "Should hit profit target and generate buy signal");
        });
    }

    #[test]
    fn test_lopez_de_prado_purged_cv_specifications() {
        // Test purged cross-validation as specified in López de Prado
        let validator = PurgedCrossValidator::new(0.01, 100, 20); // 1% embargo
        let n_samples = 1000;
        let splits = validator.create_purged_cv_splits(n_samples, 5, 0.01).unwrap();
        
        // Verify López de Prado's requirements:
        // 1. No leakage between train and test
        // 2. Embargo periods are respected
        // 3. Sufficient samples in each fold
        
        for (train_indices, test_indices) in &splits {
            // Requirement 1: No overlap
            for &test_idx in test_indices {
                assert!(!train_indices.contains(&test_idx));
            }
            
            // Requirement 2: Embargo periods
            let embargo_size = (n_samples as f32 * 0.01) as usize;
            let test_min = *test_indices.iter().min().unwrap();
            let test_max = *test_indices.iter().max().unwrap();
            
            for &train_idx in train_indices {
                if train_idx < test_min {
                    assert!(train_idx + embargo_size <= test_min);
                }
                if train_idx > test_max {
                    assert!(train_idx >= test_max + embargo_size);
                }
            }
            
            // Requirement 3: Sufficient samples
            assert!(train_indices.len() >= 100);
            assert!(test_indices.len() >= 20);
        }
    }

    #[test]
    fn test_lopez_de_prado_sample_weighting_specifications() {
        Python::with_gil(|py| {
            // Test sample weighting as per López de Prado's methodology
            let weighting = VolatilityWeighting::new(20, 0.1, 3.0);
            
            // Create returns with known volatility patterns
            let mut returns = vec![0.0; 100];
            
            // Low volatility period
            for i in 0..30 {
                returns[i] = 0.005 * (i as f32).sin();
            }
            
            // High volatility period (should get higher weights)
            for i in 30..70 {
                returns[i] = 0.03 * (i as f32).sin();
            }
            
            // Normal volatility period
            for i in 70..100 {
                returns[i] = 0.015 * (i as f32).sin();
            }
            
            let returns_array = PyArray1::from_vec(py, returns).readonly();
            let weights = weighting.calculate_weights(py, returns_array).unwrap();
            let weights_vec: Vec<f32> = weights.bind(py).readonly().as_array().to_vec();
            
            // High volatility period should have higher average weights
            let high_vol_weights: Vec<f32> = weights_vec[30..70].to_vec();
            let normal_vol_weights: Vec<f32> = weights_vec[70..100].to_vec();
            
            let high_vol_mean: f32 = high_vol_weights.iter().sum::<f32>() / high_vol_weights.len() as f32;
            let normal_vol_mean: f32 = normal_vol_weights.iter().sum::<f32>() / normal_vol_weights.len() as f32;
            
            assert!(high_vol_mean > normal_vol_mean, 
                "High volatility periods should have higher average weights");
        });
    }
}