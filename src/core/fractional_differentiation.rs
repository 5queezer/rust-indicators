use crate::core::stationarity_tests::StationarityTests;
use ndarray::{s, Array1};

/// Implements López de Prado's fractional differentiation method
/// for achieving stationarity while preserving memory.
pub struct FractionalDifferentiator {
    pub d: f64,
    pub threshold: f64,
    pub weights: Vec<f64>,
    pub window_size: usize,
}

impl FractionalDifferentiator {
    pub fn new(d: f64, threshold: f64) -> Self {
        let weights = Self::compute_weights(d, threshold);
        Self {
            d,
            threshold,
            window_size: weights.len(),
            weights,
        }
    }

    fn compute_weights(d: f64, threshold: f64) -> Vec<f64> {
        let mut weights = vec![1.0];
        let mut k = 1;

        loop {
            let weight = -weights[k - 1] * (d - k as f64 + 1.0) / k as f64;
            if weight.abs() < threshold {
                break;
            }
            weights.push(weight);
            k += 1;
        }
        weights
    }

    pub fn transform(&self, series: &Array1<f64>) -> Array1<f64> {
        let n = series.len();
        let mut result = Array1::zeros(n);

        for i in self.window_size..n {
            let start = i - self.window_size + 1;
            let window = series.slice(s![start..=i]);
            result[i] = self.weights.iter()
                .zip(window.iter().rev())
                .map(|(w, x)| w * x)
                .sum();
        }
        result
    }

    pub fn find_optimal_d(series: &Array1<f64>, confidence_level: f64) -> f64 {
        let mut low = 0.0;
        let mut high = 2.0;

        while high - low > 0.01 {
            let mid = (low + high) / 2.0;
            let differentiator = Self::new(mid, 1e-5);
            let diff_series = differentiator.transform(series);

            if StationarityTests::adf_test(&diff_series, confidence_level) {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_compute_weights() {
        let differentiator = FractionalDifferentiator::new(0.5, 1e-5);
        // Known weights for d=0.5
        let expected_weights = vec![1.0, -0.5, -0.125, -0.0625, -0.0390625];
        for (i, &w) in differentiator.weights.iter().take(5).enumerate() {
            assert_relative_eq!(w, expected_weights[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_transform_d_zero() {
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let differentiator = FractionalDifferentiator::new(0.0, 1e-5);
        let result = differentiator.transform(&series);

        // d=0 should be identity, but weights are not just [1.0].
        // The first element is 1.0, others are very small.
        // So the transformed series should be very close to original.
        for i in differentiator.window_size..series.len() {
            assert_relative_eq!(result[i], series[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_transform_d_one() {
        let series = array![1.0, 2.0, 4.0, 7.0, 11.0]; // diffs are 1, 2, 3, 4
        let differentiator = FractionalDifferentiator::new(1.0, 1e-5);
        let result = differentiator.transform(&series);

        // d=1 should be close to first difference
        let expected_diffs = array![f64::NAN, 1.0, 2.0, 3.0, 4.0];
        for i in differentiator.window_size..series.len() {
            assert_relative_eq!(result[i], expected_diffs[i], epsilon = 1e-4);
        }
    }
}
