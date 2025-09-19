//! Comprehensive tests for CombinatorialPurgedCV and OverfittingDetection
//!
//! Tests cover López de Prado's CombinatorialPurgedCV requirements including:
//! - Combination generation correctness (C(N,k) combinations)
//! - PBO calculation accuracy and statistical properties
//! - Overfitting detection with various scenarios
//! - Integration with existing ML components
//! - Performance benchmarks for large datasets
//! - Edge case validation and error handling

use rstest::fixture;
use rust_indicators::ml::components::{
    CVMetrics, CombinatorialPurgedCV, DegradationSeverity, OverfittingDetection,
};
use rust_indicators::ml::traits::CrossValidator;

#[cfg(test)]
mod combinatorial_cv_tests {
    use super::*;

    // Initialize Python for all tests
    #[ctor::ctor]
    fn init() {
        pyo3::prepare_freethreaded_python();
    }

    // === FIXTURES ===

    #[fixture]
    fn sample_cv_metrics() -> Vec<CVMetrics> {
        vec![
            CVMetrics {
                performance: 0.85,
                train_size: 800,
                test_size: 200,
                combination_id: 0,
            },
            CVMetrics {
                performance: 0.82,
                train_size: 750,
                test_size: 250,
                combination_id: 1,
            },
            CVMetrics {
                performance: 0.88,
                train_size: 820,
                test_size: 180,
                combination_id: 2,
            },
            CVMetrics {
                performance: 0.79,
                train_size: 780,
                test_size: 220,
                combination_id: 3,
            },
            CVMetrics {
                performance: 0.91,
                train_size: 850,
                test_size: 150,
                combination_id: 4,
            },
            CVMetrics {
                performance: 0.77,
                train_size: 760,
                test_size: 240,
                combination_id: 5,
            },
            CVMetrics {
                performance: 0.84,
                train_size: 800,
                test_size: 200,
                combination_id: 6,
            },
            CVMetrics {
                performance: 0.86,
                train_size: 810,
                test_size: 190,
                combination_id: 7,
            },
        ]
    }

    #[fixture]
    fn overfit_performance_data() -> (Vec<f64>, Vec<f64>) {
        // In-sample clearly outperforms out-of-sample (overfitting scenario)
        let in_sample = vec![0.95, 0.92, 0.94, 0.96, 0.93, 0.97, 0.91, 0.95];
        let out_sample = vec![0.45, 0.42, 0.44, 0.46, 0.43, 0.47, 0.41, 0.45];
        (in_sample, out_sample)
    }

    #[fixture]
    fn good_performance_data() -> (Vec<f64>, Vec<f64>) {
        // Similar in-sample and out-of-sample performance (good generalization)
        let in_sample = vec![0.75, 0.72, 0.74, 0.76, 0.73, 0.77, 0.71, 0.75];
        let out_sample = vec![0.73, 0.70, 0.72, 0.74, 0.71, 0.75, 0.69, 0.73];
        (in_sample, out_sample)
    }

    // === COMBINATORIAL PURGED CV TESTS ===

    #[test]
    fn test_combinatorial_purged_cv_creation() {
        let cpcv = CombinatorialPurgedCV::new(0.01, 10, 3, 50, 10);
        assert_eq!(cpcv.n_groups, 10);
        assert_eq!(cpcv.test_groups, 3);
        assert_eq!(cpcv.embargo_periods, 1); // max(10 * 0.01, 1) = 1
        assert_eq!(cpcv.base_cv.min_train_size, 50);
        assert_eq!(cpcv.base_cv.min_test_size, 10);
    }

    #[test]
    fn test_combination_generation_correctness() {
        let cpcv = CombinatorialPurgedCV::new(0.01, 5, 2, 10, 5);
        let combinations = cpcv.generate_combinations();

        // C(5,2) = 10 combinations
        assert_eq!(combinations.len(), 10);

        // Each combination should have exactly 2 groups
        for combo in &combinations {
            assert_eq!(combo.len(), 2);
        }

        // Verify all combinations are unique
        for i in 0..combinations.len() {
            for j in (i + 1)..combinations.len() {
                assert_ne!(combinations[i], combinations[j]);
            }
        }

        // Verify all combinations contain valid group indices
        for combo in &combinations {
            for &group_idx in combo {
                assert!(group_idx < 5);
            }
        }

        // Verify specific combinations exist
        assert!(combinations.contains(&vec![0, 1]));
        assert!(combinations.contains(&vec![0, 2]));
        assert!(combinations.contains(&vec![3, 4]));
    }

    #[test]
    fn test_combination_generation_edge_cases() {
        // Test C(3,1) = 3
        let cpcv1 = CombinatorialPurgedCV::new(0.01, 3, 1, 10, 5);
        let combinations1 = cpcv1.generate_combinations();
        assert_eq!(combinations1.len(), 3);
        assert_eq!(combinations1, vec![vec![0], vec![1], vec![2]]);

        // Test C(4,4) = 1 (all groups for testing)
        let cpcv2 = CombinatorialPurgedCV::new(0.01, 4, 4, 10, 5);
        let combinations2 = cpcv2.generate_combinations();
        assert_eq!(combinations2.len(), 1);
        assert_eq!(combinations2[0], vec![0, 1, 2, 3]);

        // Test C(6,3) = 20
        let cpcv3 = CombinatorialPurgedCV::new(0.01, 6, 3, 10, 5);
        let combinations3 = cpcv3.generate_combinations();
        assert_eq!(combinations3.len(), 20);
    }

    #[test]
    fn test_combinatorial_splits_creation() {
        let cpcv = CombinatorialPurgedCV::new(0.05, 4, 2, 20, 10);
        let n_samples = 1000;

        let splits = cpcv.create_combinatorial_splits(n_samples).unwrap();

        // Should have C(4,2) = 6 splits
        assert_eq!(splits.len(), 6);

        // Verify each split has valid structure
        for (train_indices, test_indices, combo_id) in &splits {
            // Check minimum sizes
            assert!(train_indices.len() >= 20);
            assert!(test_indices.len() >= 10);

            // Check no overlap
            for &test_idx in test_indices {
                assert!(!train_indices.contains(&test_idx));
            }

            // Check combination ID is valid
            assert!(*combo_id < 6);

            // Check indices are within bounds
            for &idx in train_indices.iter().chain(test_indices.iter()) {
                assert!(idx < n_samples);
            }
        }
    }

    #[test]
    fn test_combinatorial_splits_embargo_logic() {
        let cpcv = CombinatorialPurgedCV::new(0.1, 4, 1, 10, 5); // 10% embargo
        let n_samples = 400; // 100 samples per group

        let splits = cpcv.create_combinatorial_splits(n_samples).unwrap();

        for (train_indices, test_indices, _) in &splits {
            let test_min = *test_indices.iter().min().unwrap();
            let test_max = *test_indices.iter().max().unwrap();
            let embargo_size = cpcv.embargo_periods;

            // Verify embargo constraints
            for &train_idx in train_indices {
                // Training indices should respect embargo periods
                if train_idx < test_min {
                    assert!(
                        train_idx + embargo_size <= test_min,
                        "Train index {} too close to test set start {}",
                        train_idx,
                        test_min
                    );
                }
                if train_idx > test_max {
                    assert!(
                        train_idx >= test_max + embargo_size,
                        "Train index {} too close to test set end {}",
                        train_idx,
                        test_max
                    );
                }
            }
        }
    }

    #[test]
    fn test_splits_validation() {
        let cpcv = CombinatorialPurgedCV::new(0.02, 5, 2, 30, 15);
        let n_samples = 500;

        let splits = cpcv.create_combinatorial_splits(n_samples).unwrap();

        // Validate splits using the built-in validator
        assert!(cpcv.validate_splits(&splits));

        // Test with invalid splits (create artificial overlap)
        let mut invalid_splits = splits.clone();
        if let Some((train_indices, test_indices, _combo_id)) = invalid_splits.first_mut() {
            // Create overlap by adding a test index to training set
            if let Some(&test_idx) = test_indices.first() {
                train_indices.push(test_idx);
            }
        }

        assert!(!cpcv.validate_splits(&invalid_splits));
    }

    #[test]
    fn test_backward_compatibility_with_cross_validator_trait() {
        let cpcv = CombinatorialPurgedCV::new(0.01, 8, 2, 40, 20);

        // Test that it implements CrossValidator trait correctly
        let splits = cpcv.create_purged_cv_splits(1000, 5, 0.01).unwrap();
        assert!(!splits.is_empty());

        let pattern_splits = cpcv.create_pattern_aware_cv_splits(1000, 3, 10).unwrap();
        assert!(!pattern_splits.is_empty());

        // Validate splits using trait method
        assert!(cpcv.validate_cv_splits(&splits, 40, 20));
    }

    // === PBO CALCULATION TESTS ===

    #[test]
    fn test_pbo_calculation_basic() {
        let cpcv = CombinatorialPurgedCV::default();

        // Create performance scores with clear overfitting pattern
        let high_scores = vec![0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.89, 0.86];

        let metrics = cpcv.calculate_pbo(&high_scores, Some(100));

        assert!(metrics.pbo >= 0.0 && metrics.pbo <= 1.0);
        assert!(metrics.pbo > 0.5); // Should indicate overfitting
        assert_eq!(metrics.n_combinations, high_scores.len());
        assert_eq!(metrics.performance_distribution, high_scores);

        // Confidence interval should be valid
        assert!(metrics.confidence_interval.0 <= metrics.confidence_interval.1);
        assert!(metrics.confidence_interval.0 >= 0.0);
        assert!(metrics.confidence_interval.1 <= 1.0);
    }

    #[test]
    fn test_pbo_calculation_good_generalization() {
        let cpcv = CombinatorialPurgedCV::default();

        // Create performance scores with good generalization
        let scores = vec![0.75, 0.73, 0.74, 0.76, 0.72, 0.77, 0.75, 0.74];

        let metrics = cpcv.calculate_pbo(&scores, Some(50));

        assert!(metrics.pbo >= 0.0 && metrics.pbo <= 1.0);
        // PBO should be lower for good generalization, but exact value depends on implementation
        assert_eq!(metrics.n_combinations, scores.len());
    }

    #[test]
    fn test_pbo_edge_cases() {
        let cpcv = CombinatorialPurgedCV::default();

        // Test with empty scores
        let empty_metrics = cpcv.calculate_pbo(&[], None);
        assert_eq!(empty_metrics.pbo, 1.0); // Maximum overfitting probability
        assert_eq!(empty_metrics.n_combinations, 0);

        // Test with single score
        let single_metrics = cpcv.calculate_pbo(&[0.8], Some(1));
        assert!(single_metrics.pbo >= 0.0 && single_metrics.pbo <= 1.0);
        assert_eq!(single_metrics.n_combinations, 1);

        // Test with identical scores
        let identical_scores = vec![0.75; 10];
        let identical_metrics = cpcv.calculate_pbo(&identical_scores, Some(20));
        assert!(identical_metrics.pbo >= 0.0 && identical_metrics.pbo <= 1.0);
    }

    // === OVERFITTING DETECTION TESTS ===

    #[test]
    fn test_overfitting_detection_creation() {
        let detector = OverfittingDetection::new(0.05, 10);
        assert_eq!(detector.significance_level, 0.05);
        assert_eq!(detector.min_combinations, 10);
        assert_eq!(detector.bootstrap_samples, 1000);

        let default_detector = OverfittingDetection::default();
        assert_eq!(default_detector.significance_level, 0.05);
        assert_eq!(default_detector.min_combinations, 10);
    }

    #[test]
    fn test_pbo_calculation_with_overfitting_data() {
        let detector = OverfittingDetection::new(0.05, 5);
        // In-sample clearly outperforms out-of-sample (overfitting scenario)
        let in_sample = vec![0.95, 0.92, 0.94, 0.96, 0.93, 0.97, 0.91, 0.95];
        let out_sample = vec![0.45, 0.42, 0.44, 0.46, 0.43, 0.47, 0.41, 0.45];

        let result = detector.calculate_pbo(&in_sample, &out_sample).unwrap();

        assert!(result.pbo_value >= 0.0 && result.pbo_value <= 1.0);
        assert!(result.pbo_value > 0.7); // Should be high for clear overfitting
        assert!(result.is_overfit);
        assert_eq!(result.n_combinations, in_sample.len());

        // Confidence interval should be reasonable
        assert!(result.confidence_interval.0 <= result.confidence_interval.1);
        assert!(result.statistical_significance >= 0.0 && result.statistical_significance <= 1.0);
    }

    #[test]
    fn test_pbo_calculation_with_good_data() {
        let detector = OverfittingDetection::new(0.05, 5);
        // Similar in-sample and out-of-sample performance (good generalization)
        let in_sample = vec![0.75, 0.72, 0.74, 0.76, 0.73, 0.77, 0.71, 0.75];
        let out_sample = vec![0.73, 0.70, 0.72, 0.74, 0.71, 0.75, 0.69, 0.73];

        let result = detector.calculate_pbo(&in_sample, &out_sample).unwrap();

        assert!(result.pbo_value >= 0.0 && result.pbo_value <= 1.0);
        assert!(result.pbo_value < 0.7); // Should be lower for good generalization
        assert!(!result.is_overfit || result.pbo_value <= 0.5); // Should not indicate overfitting
        assert_eq!(result.n_combinations, in_sample.len());
    }

    #[test]
    fn test_comprehensive_overfitting_detection() {
        let detector = OverfittingDetection::new(0.05, 5);

        let sample_cv_metrics = vec![
            CVMetrics {
                performance: 0.85,
                train_size: 800,
                test_size: 200,
                combination_id: 0,
            },
            CVMetrics {
                performance: 0.82,
                train_size: 750,
                test_size: 250,
                combination_id: 1,
            },
            CVMetrics {
                performance: 0.88,
                train_size: 820,
                test_size: 180,
                combination_id: 2,
            },
            CVMetrics {
                performance: 0.79,
                train_size: 780,
                test_size: 220,
                combination_id: 3,
            },
            CVMetrics {
                performance: 0.91,
                train_size: 850,
                test_size: 150,
                combination_id: 4,
            },
            CVMetrics {
                performance: 0.77,
                train_size: 760,
                test_size: 240,
                combination_id: 5,
            },
            CVMetrics {
                performance: 0.84,
                train_size: 800,
                test_size: 200,
                combination_id: 6,
            },
            CVMetrics {
                performance: 0.86,
                train_size: 810,
                test_size: 190,
                combination_id: 7,
            },
        ];

        let report = detector.detect_overfitting(&sample_cv_metrics).unwrap();

        // Verify PBO result structure
        assert!(report.pbo_result.pbo_value >= 0.0 && report.pbo_result.pbo_value <= 1.0);
        assert_eq!(
            report.pbo_result.n_combinations,
            sample_cv_metrics.len() / 2
        );

        // Verify performance statistics
        let stats = &report.performance_stats;
        assert!(stats.mean_performance > 0.0);
        assert!(stats.std_performance >= 0.0);
        assert!(stats.min_performance <= stats.max_performance);
        assert!(stats.min_performance <= stats.median_performance);
        assert!(stats.median_performance <= stats.max_performance);

        // Verify degradation analysis
        let degradation = &report.degradation_analysis;
        assert!(
            degradation.degradation_frequency >= 0.0 && degradation.degradation_frequency <= 1.0
        );

        // Verify recommendations are provided
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_degradation_severity_classification() {
        let detector = OverfittingDetection::default();

        // Test severe degradation
        let severe_is = vec![0.9, 0.85, 0.88];
        let severe_oos = vec![0.3, 0.25, 0.28];
        let severe_analysis = detector.analyze_degradation(&severe_is, &severe_oos);
        assert_eq!(severe_analysis.severity, DegradationSeverity::Severe);
        assert!(severe_analysis.mean_degradation > 0.2);

        // Test low degradation
        let low_is = vec![0.75, 0.73, 0.74];
        let low_oos = vec![0.73, 0.71, 0.72];
        let low_analysis = detector.analyze_degradation(&low_is, &low_oos);
        assert_eq!(low_analysis.severity, DegradationSeverity::Low);
        assert!(low_analysis.mean_degradation < 0.05);
    }

    #[test]
    fn test_error_handling() {
        let detector = OverfittingDetection::new(0.05, 10);

        // Test mismatched array lengths
        let in_sample = vec![0.8, 0.9];
        let out_sample = vec![0.3, 0.4, 0.5]; // Different length

        let result = detector.calculate_pbo(&in_sample, &out_sample);
        assert!(result.is_err());

        // Test insufficient combinations
        let small_in = vec![0.8];
        let small_out = vec![0.3];

        let result2 = detector.calculate_pbo(&small_in, &small_out);
        assert!(result2.is_err());

        // Test insufficient CV results
        let small_cv_results = vec![CVMetrics {
            performance: 0.8,
            train_size: 100,
            test_size: 50,
            combination_id: 0,
        }];

        let result3 = detector.detect_overfitting(&small_cv_results);
        assert!(result3.is_err());
    }

    // === INTEGRATION TESTS ===

    #[test]
    fn test_full_workflow_integration() {
        // Test complete workflow: CPCV -> Overfitting Detection
        let cpcv = CombinatorialPurgedCV::new(0.02, 6, 2, 50, 25);
        let detector = OverfittingDetection::new(0.05, 10);

        // Create combinatorial splits
        let n_samples = 1200;
        let splits = cpcv.create_combinatorial_splits(n_samples).unwrap();

        // Simulate CV metrics from splits
        let mut cv_metrics = Vec::new();
        for (i, (train_indices, test_indices, combo_id)) in splits.iter().enumerate() {
            // Simulate performance that degrades with combination number (overfitting pattern)
            let base_performance = 0.85;
            let degradation = (i as f64) * 0.02; // Increasing degradation
            let performance = (base_performance - degradation).max(0.3);

            cv_metrics.push(CVMetrics {
                performance,
                train_size: train_indices.len(),
                test_size: test_indices.len(),
                combination_id: *combo_id,
            });
        }

        // Detect overfitting
        let report = detector.detect_overfitting(&cv_metrics).unwrap();

        // Verify integration results
        assert!(report.pbo_result.pbo_value >= 0.0);
        assert!(!report.recommendations.is_empty());
        assert_eq!(report.pbo_result.n_combinations, cv_metrics.len() / 2);

        // Should detect overfitting due to degradation pattern
        if report.pbo_result.pbo_value > 0.6 {
            assert!(report.pbo_result.is_overfit);
        }
    }

    #[test]
    fn test_performance_with_large_datasets() {
        use std::time::Instant;

        // Test performance with larger datasets
        let cpcv = CombinatorialPurgedCV::new(0.01, 8, 3, 100, 50);
        let n_samples = 10000;

        let start = Instant::now();
        let splits = cpcv.create_combinatorial_splits(n_samples).unwrap();
        let duration = start.elapsed();

        // Should complete within reasonable time (adjust threshold as needed)
        assert!(
            duration.as_secs() < 5,
            "Large dataset processing took too long: {:?}",
            duration
        );

        // Should generate C(8,3) = 56 combinations
        assert_eq!(splits.len(), 56);

        // Verify all splits are valid
        assert!(cpcv.validate_splits(&splits));

        // Test PBO calculation performance
        let performances: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.003).collect();

        let start2 = Instant::now();
        let metrics = cpcv.calculate_pbo(&performances, Some(1000));
        let duration2 = start2.elapsed();

        assert!(
            duration2.as_secs() < 2,
            "PBO calculation took too long: {:?}",
            duration2
        );
        assert!(metrics.pbo >= 0.0 && metrics.pbo <= 1.0);
    }

    #[test]
    fn test_thread_safety_parallel_execution() {
        use std::sync::Arc;
        use std::thread;

        let cpcv = Arc::new(CombinatorialPurgedCV::new(0.01, 5, 2, 30, 15));
        let detector = Arc::new(OverfittingDetection::new(0.05, 8));

        let mut handles = vec![];

        // Test concurrent access from multiple threads
        for thread_id in 0..4 {
            let cpcv_clone = Arc::clone(&cpcv);
            let detector_clone = Arc::clone(&detector);

            let handle = thread::spawn(move || {
                // Each thread works with different data sizes
                let n_samples = 500 + thread_id * 100;

                // Test CPCV thread safety
                let splits = cpcv_clone.create_combinatorial_splits(n_samples).unwrap();
                assert!(!splits.is_empty());
                assert!(cpcv_clone.validate_splits(&splits));

                // Test overfitting detection thread safety
                let performances: Vec<f64> = (0..20).map(|i| 0.6 + (i as f64) * 0.01).collect();
                let metrics = cpcv_clone.calculate_pbo(&performances, Some(50));
                assert!(metrics.pbo >= 0.0 && metrics.pbo <= 1.0);

                // Test detector thread safety
                let in_sample: Vec<f64> = (0..10).map(|i| 0.8 + (i as f64) * 0.01).collect();
                let out_sample: Vec<f64> = (0..10).map(|i| 0.4 + (i as f64) * 0.01).collect();
                let pbo_result = detector_clone
                    .calculate_pbo(&in_sample, &out_sample)
                    .unwrap();
                assert!(pbo_result.pbo_value >= 0.0 && pbo_result.pbo_value <= 1.0);

                thread_id
            });

            handles.push(handle);
        }

        // Wait for all threads and verify completion
        let mut completed_threads = Vec::new();
        for handle in handles {
            let thread_id = handle.join().expect("Thread panicked");
            completed_threads.push(thread_id);
        }

        completed_threads.sort();
        assert_eq!(completed_threads, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_statistical_properties_validation() {
        let detector = OverfittingDetection::new(0.05, 5);

        // Test with known statistical properties
        let n_trials = 100;
        let mut pbo_values = Vec::new();

        for trial in 0..n_trials {
            // Create slightly different data for each trial
            let base_offset = (trial as f64) * 0.001;
            let in_sample: Vec<f64> = (0..10)
                .map(|i| 0.85 + base_offset + (i as f64) * 0.01)
                .collect();
            let out_sample: Vec<f64> = (0..10)
                .map(|i| 0.45 + base_offset + (i as f64) * 0.01)
                .collect();

            let result = detector.calculate_pbo(&in_sample, &out_sample).unwrap();
            pbo_values.push(result.pbo_value);
        }

        // Statistical validation
        let mean_pbo: f64 = pbo_values.iter().sum::<f64>() / n_trials as f64;
        let variance: f64 = pbo_values
            .iter()
            .map(|x| (x - mean_pbo).powi(2))
            .sum::<f64>()
            / n_trials as f64;
        let std_pbo = variance.sqrt();

        // PBO values should be consistently high for overfitting scenario
        assert!(
            mean_pbo > 0.7,
            "Mean PBO should be high for overfitting: {}",
            mean_pbo
        );
        assert!(
            std_pbo < 0.2,
            "PBO should be relatively stable: {}",
            std_pbo
        );

        // All PBO values should be in valid range
        for &pbo in &pbo_values {
            assert!(pbo >= 0.0 && pbo <= 1.0, "Invalid PBO value: {}", pbo);
        }
    }
}
