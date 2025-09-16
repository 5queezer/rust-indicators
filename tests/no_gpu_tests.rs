mod no_gpu_tests {
    use rust_indicators::cpu_impls::vpin_cpu_kernel;
    use rand::Rng;

    fn generate_random_data(len: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen_range(0.0..100.0)).collect()
    }

    fn assert_approx_eq(a: &[f64], b: &[f64], epsilon: f64) {
        assert_eq!(a.len(), b.len(), "Vectors have different lengths");
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < epsilon,
                "Mismatch at index {}: expected {}, got {}", i, b[i], a[i]);
        }
    }

    #[test]
    fn test_vpin_cpu_basic_calculation() {
        let buy_vols = [100.0, 200.0, 300.0];
        let sell_vols = [50.0, 100.0, 150.0];
        let window = 2;
        let expected = vec![0.0, 0.0, 1.0 / 3.0];
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_approx_eq(&result, &expected, 1e-9);
    }

    #[test]
    fn test_vpin_cpu_balanced_volumes() {
        let buy_vols = [10.0, 10.0, 10.0];
        let sell_vols = [10.0, 10.0, 10.0];
        let window = 2;
        let expected = vec![0.0, 0.0, 0.0];
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_approx_eq(&result, &expected, 1e-9);
    }

    #[test]
    fn test_vpin_cpu_extreme_imbalance() {
        let buy_vols = [10.0, 20.0, 30.0];
        let sell_vols = [0.0, 0.0, 0.0];
        let window = 2;
        let expected = vec![0.0, 0.0, 1.0];
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_approx_eq(&result, &expected, 1e-9);
    }

    #[test]
    fn test_vpin_cpu_empty_arrays() {
        let empty_vols: Vec<f64> = vec![];
        let empty_result = vpin_cpu_kernel(&empty_vols, &empty_vols, 1);
        assert_eq!(empty_result, Vec::<f64>::new());
    }

    #[test]
    fn test_vpin_cpu_window_larger_than_data() {
        let small_buy = [10.0, 20.0];
        let small_sell = [5.0, 10.0];
        let large_window = 5;
        let small_result = vpin_cpu_kernel(&small_buy, &small_sell, large_window);
        assert_eq!(small_result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_vpin_cpu_zero_protection() {
        let buy_vols = [0.0, 0.0, 10.0];
        let sell_vols = [0.0, 0.0, 5.0];
        let window = 2;
        let expected = vec![0.0, 0.0, 1.0 / 3.0];
        let result = vpin_cpu_kernel(&buy_vols, &sell_vols, window);
        assert_approx_eq(&result, &expected, 1e-9);
    }

    #[test]
    fn test_vpin_cpu_random_data_properties() {
        let data_len = 1000;
        let random_buy = generate_random_data(data_len);
        let random_sell = generate_random_data(data_len);
        let random_window = 20;
        let random_result = vpin_cpu_kernel(&random_buy, &random_sell, random_window);
        for (i, &val) in random_result.iter().enumerate() {
            if i >= random_window - 1 {
                assert!(val >= 0.0 && val <= 1.0, "Value {} at index {} out of range [0,1]", val, i);
            } else {
                assert_eq!(val, 0.0, "Initial value {} at index {} should be 0.0", val, i);
            }
        }
    }
}