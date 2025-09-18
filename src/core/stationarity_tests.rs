use ndarray::{s, Array1};
use nalgebra::{DMatrix, DVector};

/// A suite of tests for checking the stationarity of a time series.
pub struct StationarityTests;

impl StationarityTests {
    pub fn adf_test(series: &Array1<f64>, confidence_level: f64) -> bool {
        if series.len() < 20 {
            return false; // Need minimum samples for meaningful test
        }

        let adf_stat = Self::calculate_adf_statistic(series);
        let critical_value = Self::get_adf_critical_value(confidence_level, series.len());

        // ADF test: reject null hypothesis (non-stationary) if statistic < critical value
        adf_stat < critical_value
    }

    fn calculate_adf_statistic(series: &Array1<f64>) -> f64 {
        let n = series.len();
        let y = series.slice(s![1..]).to_owned(); // y_t
        let y_lag = series.slice(s![..n-1]).to_owned(); // y_{t-1}
        let delta_y: Array1<f64> = &y - &y_lag; // Δy_t

        // Simple ADF regression: Δy_t = α + β*y_{t-1} + ε_t
        // We test H0: β = 0 (unit root) vs H1: β < 0 (stationary)

        let x_matrix = Self::build_design_matrix(&y_lag);
        let beta_hat = Self::ols_regression(&x_matrix, &delta_y);

        if beta_hat.len() < 2 {
            return 0.0; // Regression failed
        }

        let beta_coefficient = beta_hat[1]; // Coefficient on y_{t-1}
        let se_beta = Self::calculate_standard_error(&x_matrix, &delta_y, &beta_hat, 1);

        // t-statistic for β = 0
        if se_beta > 0.0 {
            beta_coefficient / se_beta
        } else {
            0.0
        }
    }

    fn build_design_matrix(y_lag: &Array1<f64>) -> DMatrix<f64> {
        let n = y_lag.len();
        let mut x = DMatrix::zeros(n, 2);

        // First column: intercept (all ones)
        x.column_mut(0).fill(1.0);

        // Second column: lagged y values
        for i in 0..n {
            x[(i, 1)] = y_lag[i];
        }

        x
    }

    fn ols_regression(x: &DMatrix<f64>, y: &Array1<f64>) -> Vec<f64> {
        // OLS: β = (X'X)^(-1)X'y
        let x_t = x.transpose();
        let xtx = &x_t * x;

        if let Some(xtx_inv) = xtx.try_inverse() {
            let xty = x_t * DVector::from_column_slice(y.as_slice().unwrap());
            let beta = xtx_inv * xty;
            beta.as_slice().to_vec()
        } else {
            vec![0.0; x.ncols()] // Return zeros if singular
        }
    }

    fn calculate_standard_error(
        x: &DMatrix<f64>,
        y: &Array1<f64>,
        beta: &[f64],
        coef_index: usize
    ) -> f64 {
        let n = x.nrows();
        let k = x.ncols();

        if n <= k {
            return 1.0; // Not enough observations
        }

        // Calculate residuals
        let mut residuals = vec![0.0; n];
        for i in 0..n {
            let predicted: f64 = (0..k).map(|j| x[(i, j)] * beta[j]).sum();
            residuals[i] = y[i] - predicted;
        }

        // Residual sum of squares
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let mse = rss / (n - k) as f64;

        // Standard error of coefficient
        let x_t = x.transpose();
        let xtx = &x_t * x;

        if let Some(xtx_inv) = xtx.try_inverse() {
            (mse * xtx_inv[(coef_index, coef_index)]).sqrt()
        } else {
            1.0 // Fallback
        }
    }

    fn get_adf_critical_value(confidence_level: f64, sample_size: usize) -> f64 {
        // MacKinnon critical values (approximation)
        // These are for the case with intercept but no trend

        let base_critical = match confidence_level {
            level if level >= 0.99 => -3.43,  // 1% level
            level if level >= 0.95 => -2.86,  // 5% level
            level if level >= 0.90 => -2.57,  // 10% level
            _ => -2.57,
        };

        // Adjust for sample size (larger samples have more negative critical values)
        let size_adjustment = if sample_size > 100 {
            -0.1
        } else if sample_size > 50 {
            0.0
        } else {
            0.1
        };

        base_critical + size_adjustment
    }

    pub fn kpss_test(_series: &Array1<f64>) -> (f64, f64, bool) {
        // Kwiatkowski-Phillips-Schmidt-Shin test
        todo!()
    }

    pub fn phillips_perron_test(_series: &Array1<f64>) -> (f64, f64, bool) {
        // Phillips-Perron unit root test
        todo!()
    }
}
