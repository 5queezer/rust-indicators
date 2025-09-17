mod no_gpu_tests {
    use rand::Rng;
    use rstest::{fixture, rstest};
    use rust_indicators::backends::cpu::implementations::vpin_cpu_kernel;

    // Shared fixtures for common test data
    #[fixture]
    fn small_test_data() -> (Vec<f64>, Vec<f64>, usize) {
        (vec![10.0, 20.0, 30.0], vec![5.0, 10.0, 15.0], 2)
    }

    #[fixture]
    fn balanced_test_data() -> (Vec<f64>, Vec<f64>, usize) {
        (vec![10.0, 10.0, 10.0], vec![10.0, 10.0, 10.0], 2)
    }

    #[fixture]
    fn empty_test_data() -> (Vec<f64>, Vec<f64>, usize) {
        (vec![], vec![], 1)
    }

    fn assert_approx_eq(a: &[f64], b: &[f64], epsilon: f64) {
        assert_eq!(a.len(), b.len(), "Vectors have different lengths");
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < epsilon,
                "Mismatch at index {}: expected {}, got {}",
                i,
                b[i],
                a[i]
            );
        }
    }

    // Parameterized VPIN test cases
    #[rstest]
    #[case::basic_calculation(
        vec![100.0, 200.0, 300.0],
        vec![50.0, 100.0, 150.0],
        2,
        vec![0.0, 0.0, 1.0 / 3.0]
    )]
    #[case::balanced_volumes(
        vec![10.0, 10.0, 10.0],
        vec![10.0, 10.0, 10.0],
        2,
        vec![0.0, 0.0, 0.0]
    )]
    #[case::extreme_imbalance(
        vec![10.0, 20.0, 30.0],
        vec![0.0, 0.0, 0.0],
        2,
        vec![0.0, 0.0, 1.0]
    )]
    #[case::zero_protection(
        vec![0.0, 0.0, 10.0],
        vec![0.0, 0.0, 5.0],
        2,
        vec![0.0, 0.0, 1.0 / 3.0]
    )]
    fn test_vpin_cpu_calculations(
        #[case] buy_vols: Vec<f64>,
        #[case] sell_vols: Vec<f64>,
        #[case] window: usize,
        #[case] expected: Vec<f64>,
    ) {
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_approx_eq(&result, &expected, 1e-9);
    }

    #[rstest]
    #[case::empty_arrays(vec![], vec![], 1, vec![])]
    #[case::window_larger_than_data(vec![10.0, 20.0], vec![5.0, 10.0], 5, vec![0.0, 0.0])]
    fn test_vpin_cpu_edge_cases(
        #[case] buy_vols: Vec<f64>,
        #[case] sell_vols: Vec<f64>,
        #[case] window: usize,
        #[case] expected: Vec<f64>,
    ) {
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vpin_cpu_random_data_properties() {
        let data_len = 1000;
        let mut rng = rand::thread_rng();
        let random_buy: Vec<f64> = (0..data_len).map(|_| rng.gen_range(0.0..100.0)).collect();
        let random_sell: Vec<f64> = (0..data_len).map(|_| rng.gen_range(0.0..100.0)).collect();
        let random_window = 20;
        let random_result = vpin_cpu_kernel(&random_buy, &random_sell, random_window);

        for (i, &val) in random_result.iter().enumerate() {
            if i >= random_window - 1 {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Value {} at index {} out of range [0,1]",
                    val,
                    i
                );
            } else {
                assert_eq!(
                    val, 0.0,
                    "Initial value {} at index {} should be 0.0",
                    val, i
                );
            }
        }
    }
}
