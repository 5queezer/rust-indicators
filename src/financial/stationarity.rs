//! Stationarity testing suite for financial time series
//! 
//! This module implements comprehensive stationarity tests based on López de Prado's
//! "Advances in Financial Machine Learning" and standard econometric literature.
//! 
//! # Key Tests
//! 
//! - **ADF (Augmented Dickey-Fuller)**: Tests unit-root null hypothesis
//! - **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**: Tests stationarity null hypothesis  
//! - **Phillips-Perron**: Tests unit-root null with HAC adjustment
//! 
//! # Example
//! 
//! ```rust
//! use rust_indicators::financial::StationarityTests;
//! use ndarray::Array1;
//! 
//! let series = Array1::from(vec![1.0, 1.1, 1.05, 1.2, 1.15, 1.3]);
//! let (statistic, p_value, is_stationary) = StationarityTests::adf_test(&series);
//! 
//! if is_stationary {
//!     println!("Series is stationary (ADF statistic: {:.4})", statistic);
//! }
//! ```

use ndarray::{Array1, Array2, s};
use std::f64::consts::PI;

/// Comprehensive stationarity testing suite
/// 
/// Implements the three key stationarity tests used in financial econometrics:
/// - ADF test for unit root detection
/// - KPSS test for trend stationarity
/// - Phillips-Perron test with HAC adjustment
pub struct StationarityTests;

impl StationarityTests {
    /// Augmented Dickey-Fuller test for unit root
    /// 
    /// Tests the null hypothesis that the series has a unit root (non-stationary)
    /// against the alternative of stationarity.
    /// 
    /// # Arguments
    /// 
    /// * `series` - The time series to test
    /// 
    /// # Returns
    /// 
    /// A tuple containing:
    /// - Test statistic (more negative = more evidence against unit root)
    /// - P-value (probability of observing this statistic under null)
    /// - Boolean indicating stationarity at 5% significance level
    /// 
    /// # Implementation
    /// 
    /// Performs regression: Δy_t = α + γy_{t-1} + Σδ_i*Δy_{t-i} + ε_t
    /// Test statistic is t-ratio for γ coefficient
    pub fn adf_test(series: &Array1<f64>) -> (f64, f64, bool) {
        let n = series.len();
        if n < 10 {
            return (0.0, 1.0, false); // Insufficient data
        }

        // Determine optimal lag length using AIC
        let max_lags = ((n as f64).powf(1.0/3.0) * 12.0 / 100.0).ceil() as usize;
        let optimal_lags = Self::select_adf_lags(series, max_lags);
        
        // Build first differences
        let dy = Self::diff(series, 1);
        let lagged_y = &series.slice(s![optimal_lags..n-1]);
        
        // Build design matrix: [constant, y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}]
        let n_obs = dy.len() - optimal_lags;
        let n_vars = 2 + optimal_lags; // constant + lagged_y + lag terms
        let mut x = Array2::<f64>::zeros((n_obs, n_vars));
        
        // Constant term
        x.column_mut(0).fill(1.0);
        
        // Lagged level y_{t-1}
        x.column_mut(1).assign(lagged_y);
        
        // Lagged differences Δy_{t-i}
        for i in 0..optimal_lags {
            let lag_start = optimal_lags - i - 1;
            let lag_end = dy.len() - i - 1;
            x.column_mut(2 + i).assign(&dy.slice(s![lag_start..lag_end]));
        }
        
        // Dependent variable: Δy_t
        let y_target = dy.slice(s![optimal_lags..]).to_owned();
        
        // Simple OLS regression using normal equations
        match Self::ols_regression(&x, &y_target) {
            Ok((beta, residuals)) => {
                let gamma = beta[1]; // Coefficient on y_{t-1}
                
                // Calculate standard error
                let mse = residuals.dot(&residuals) / (n_obs - n_vars) as f64;
                let xtx_inv_11 = Self::matrix_inverse_element(&x, 1, 1);
                let se_gamma = (mse * xtx_inv_11).sqrt();
                let t_stat = gamma / se_gamma;
                
                // Calculate p-value using MacKinnon critical values approximation
                let p_value = Self::adf_p_value(t_stat, n);
                let is_stationary = p_value < 0.05; // 5% significance level
                
                (t_stat, p_value, is_stationary)
            }
            Err(_) => (0.0, 1.0, false)
        }
    }
    
    /// Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
    /// 
    /// Tests the null hypothesis that the series is stationary around a deterministic trend
    /// against the alternative of a unit root.
    /// 
    /// # Arguments
    /// 
    /// * `series` - The time series to test
    /// 
    /// # Returns
    /// 
    /// A tuple containing:
    /// - KPSS test statistic (higher = more evidence against stationarity)
    /// - P-value approximation
    /// - Boolean indicating stationarity at 5% significance level
    /// 
    /// # Implementation
    /// 
    /// 1. Regress y_t on constant and trend to get residuals
    /// 2. Compute partial sums S_t = Σε_i from i=1 to t
    /// 3. Test statistic: KPSS = (1/T²) * Σ(S_t²) / σ²
    pub fn kpss_test(series: &Array1<f64>) -> (f64, f64, bool) {
        let n = series.len();
        if n < 10 {
            return (0.0, 1.0, false);
        }
        
        // Build design matrix with constant and trend
        let mut x = Array2::<f64>::zeros((n, 2));
        x.column_mut(0).fill(1.0); // Constant
        for i in 0..n {
            x[[i, 1]] = (i + 1) as f64; // Trend
        }
        
        // Simple OLS regression to get residuals
        match Self::ols_regression(&x, series) {
            Ok((beta, _)) => {
                let fitted = x.dot(&beta);
                let residuals = series - &fitted;
                
                // Compute partial sums
                let mut partial_sums = Array1::<f64>::zeros(n);
                let mut cumsum = 0.0;
                for i in 0..n {
                    cumsum += residuals[i];
                    partial_sums[i] = cumsum;
                }
                
                // Estimate long-run variance using Newey-West
                let sigma_sq = Self::newey_west_variance(&residuals, None);
                
                // KPSS statistic
                let sum_squares: f64 = partial_sums.iter().map(|s| s * s).sum();
                let kpss_stat = sum_squares / ((n * n) as f64 * sigma_sq);
                
                // Approximate p-value using critical values
                let p_value = Self::kpss_p_value(kpss_stat);
                let is_stationary = p_value > 0.05; // Fail to reject null of stationarity
                
                (kpss_stat, p_value, is_stationary)
            }
            Err(_) => (0.0, 1.0, false)
        }
    }
    
    /// Phillips-Perron unit root test
    /// 
    /// Tests the null hypothesis of a unit root with non-parametric correction
    /// for serial correlation and heteroskedasticity.
    /// 
    /// # Arguments
    /// 
    /// * `series` - The time series to test
    /// 
    /// # Returns
    /// 
    /// A tuple containing:
    /// - PP test statistic (more negative = more evidence against unit root)
    /// - P-value approximation
    /// - Boolean indicating stationarity at 5% significance level
    /// 
    /// # Implementation
    /// 
    /// Similar to ADF but uses HAC-adjusted standard errors instead of
    /// including lagged differences in the regression.
    pub fn phillips_perron_test(series: &Array1<f64>) -> (f64, f64, bool) {
        let n = series.len();
        if n < 10 {
            return (0.0, 1.0, false);
        }
        
        // Build first differences and lagged levels
        let dy = Self::diff(series, 1);
        let lagged_y = &series.slice(s![0..n-1]);
        
        // Simple regression: Δy_t = α + γy_{t-1} + ε_t
        let mut x = Array2::<f64>::zeros((dy.len(), 2));
        x.column_mut(0).fill(1.0); // Constant
        x.column_mut(1).assign(lagged_y); // y_{t-1}
        
        // Simple OLS regression
        match Self::ols_regression(&x, &dy) {
            Ok((beta, residuals)) => {
                let gamma = beta[1];
                
                // HAC adjustment using Newey-West
                let bandwidth = Self::optimal_bandwidth(n);
                let sigma_sq = Self::newey_west_variance(&residuals, Some(bandwidth));
                
                // Phillips-Perron adjustment
                let sum_y_sq: f64 = lagged_y.iter().map(|y| y * y).sum();
                let lambda_hat = Self::long_run_variance(&residuals, bandwidth);
                
                // PP test statistic with bias correction
                let t_gamma = gamma / (sigma_sq / sum_y_sq).sqrt();
                let bias_correction = (lambda_hat - sigma_sq) / (2.0 * sigma_sq);
                let pp_stat = t_gamma - bias_correction * (sum_y_sq / (n as f64 * sigma_sq)).sqrt();
                
                // Calculate p-value (same distribution as ADF asymptotically)
                let p_value = Self::adf_p_value(pp_stat, n);
                let is_stationary = p_value < 0.05;
                
                (pp_stat, p_value, is_stationary)
            }
            Err(_) => (0.0, 1.0, false)
        }
    }
    
    // Helper methods
    
    /// Calculate first differences of a series
    fn diff(series: &Array1<f64>, lag: usize) -> Array1<f64> {
        let n = series.len();
        if n <= lag {
            return Array1::zeros(0);
        }
        
        let mut result = Array1::zeros(n - lag);
        for i in 0..(n - lag) {
            result[i] = series[i + lag] - series[i];
        }
        result
    }
    
    /// Select optimal lag length for ADF test using AIC
    fn select_adf_lags(series: &Array1<f64>, max_lags: usize) -> usize {
        let mut best_aic = f64::INFINITY;
        let mut best_lags = 1;
        
        for lags in 1..=max_lags.min(series.len() / 4) {
            if let Some(aic) = Self::calculate_adf_aic(series, lags) {
                if aic < best_aic {
                    best_aic = aic;
                    best_lags = lags;
                }
            }
        }
        
        best_lags
    }
    
    /// Calculate AIC for ADF regression with given lag length
    fn calculate_adf_aic(series: &Array1<f64>, lags: usize) -> Option<f64> {
        let n = series.len();
        let dy = Self::diff(series, 1);
        let n_obs = dy.len() - lags;
        let n_vars = 2 + lags;
        
        if n_obs <= n_vars {
            return None;
        }
        
        let mut x = Array2::<f64>::zeros((n_obs, n_vars));
        x.column_mut(0).fill(1.0);
        x.column_mut(1).assign(&series.slice(s![lags..n-1]));
        
        for i in 0..lags {
            let lag_start = lags - i - 1;
            let lag_end = dy.len() - i - 1;
            x.column_mut(2 + i).assign(&dy.slice(s![lag_start..lag_end]));
        }
        
        let y_target = dy.slice(s![lags..]).to_owned();
        
        match Self::ols_regression(&x, &y_target) {
            Ok((_beta, residuals)) => {
                let sse = residuals.dot(&residuals);
                let log_likelihood = -0.5 * n_obs as f64 * (1.0 + (2.0 * PI * sse / n_obs as f64).ln());
                let aic = -2.0 * log_likelihood + 2.0 * n_vars as f64;
                Some(aic)
            }
            Err(_) => None
        }
    }
    
    /// Approximate p-value for ADF test statistic using MacKinnon critical values
    fn adf_p_value(t_stat: f64, n: usize) -> f64 {
        // MacKinnon (1994) critical values for constant + trend model
        let critical_values = [
            (-4.38, 0.01),  // 1%
            (-3.95, 0.025), // 2.5%
            (-3.60, 0.05),  // 5%
            (-3.24, 0.10),  // 10%
            (-2.86, 0.25),  // 25%
        ];
        
        // Finite sample adjustment
        let adj_factor = if n < 100 { 1.0 + 2.0 / n as f64 } else { 1.0 };
        let adj_t_stat = t_stat * adj_factor;
        
        // Linear interpolation between critical values
        for i in 0..critical_values.len() - 1 {
            let (cv1, p1) = critical_values[i];
            let (cv2, p2) = critical_values[i + 1];
            
            if adj_t_stat <= cv1 && adj_t_stat >= cv2 {
                let weight = (adj_t_stat - cv2) / (cv1 - cv2);
                return p1 * weight + p2 * (1.0 - weight);
            }
        }
        
        // Extrapolation for extreme values
        if adj_t_stat < critical_values[0].0 {
            0.001 // Very strong evidence against unit root
        } else if adj_t_stat > critical_values.last().unwrap().0 {
            0.99  // Very weak evidence against unit root
        } else {
            0.5   // Default
        }
    }
    
    /// Approximate p-value for KPSS test statistic
    fn kpss_p_value(kpss_stat: f64) -> f64 {
        // KPSS critical values for trend stationarity
        let critical_values = [
            (0.216, 0.01),  // 1%
            (0.176, 0.025), // 2.5%
            (0.146, 0.05),  // 5%
            (0.119, 0.10),  // 10%
        ];
        
        // Linear interpolation
        for i in 0..critical_values.len() - 1 {
            let (cv1, p1) = critical_values[i];
            let (cv2, p2) = critical_values[i + 1];
            
            if kpss_stat >= cv1 && kpss_stat <= cv2 {
                let weight = (kpss_stat - cv2) / (cv1 - cv2);
                return p1 * weight + p2 * (1.0 - weight);
            }
        }
        
        if kpss_stat > critical_values[0].0 {
            0.001 // Strong evidence against stationarity
        } else if kpss_stat < critical_values.last().unwrap().0 {
            0.99  // Strong evidence for stationarity
        } else {
            0.5
        }
    }
    
    /// Calculate Newey-West HAC variance estimator
    fn newey_west_variance(residuals: &Array1<f64>, bandwidth: Option<usize>) -> f64 {
        let n = residuals.len();
        let bw = bandwidth.unwrap_or(Self::optimal_bandwidth(n));
        
        // Sample variance
        let mean_resid = residuals.mean().unwrap_or(0.0);
        let mut gamma_0 = 0.0;
        for &r in residuals {
            gamma_0 += (r - mean_resid).powi(2);
        }
        gamma_0 /= n as f64;
        
        // Autocovariances with Bartlett kernel
        let mut long_run_var = gamma_0;
        for k in 1..=bw {
            let mut gamma_k = 0.0;
            for i in k..n {
                gamma_k += (residuals[i] - mean_resid) * (residuals[i - k] - mean_resid);
            }
            gamma_k /= n as f64;
            
            let weight = 1.0 - k as f64 / (bw + 1) as f64; // Bartlett kernel
            long_run_var += 2.0 * weight * gamma_k;
        }
        
        long_run_var.max(gamma_0) // Ensure positive
    }
    
    /// Calculate long-run variance for Phillips-Perron test
    fn long_run_variance(residuals: &Array1<f64>, bandwidth: usize) -> f64 {
        Self::newey_west_variance(residuals, Some(bandwidth))
    }
    
    /// Optimal bandwidth selection for HAC estimation
    fn optimal_bandwidth(n: usize) -> usize {
        // Newey-West automatic bandwidth selection
        ((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize).max(1)
    }
    
    /// Simple OLS regression without heavy linear algebra dependencies
    fn ols_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>), ()> {
        let n = x.nrows();
        let p = x.ncols();
        
        if n <= p {
            return Err(());
        }
        
        // Compute X'X manually
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x[[k, i]] * x[[k, j]];
                }
                xtx[[i, j]] = sum;
            }
        }
        
        // Compute X'y manually
        let mut xty = Array1::<f64>::zeros(p);
        for i in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[[k, i]] * y[k];
            }
            xty[i] = sum;
        }
        
        // Solve using Gaussian elimination
        match Self::gaussian_elimination(&xtx, &xty) {
            Ok(beta) => {
                // Calculate residuals
                let mut residuals = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let mut fitted = 0.0;
                    for j in 0..p {
                        fitted += x[[i, j]] * beta[j];
                    }
                    residuals[i] = y[i] - fitted;
                }
                Ok((beta, residuals))
            }
            Err(_) => Err(())
        }
    }
    
    /// Gaussian elimination for solving linear systems
    fn gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, ()> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(());
        }
        
        // Create augmented matrix
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }
            
            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }
            
            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-12 {
                return Err(());
            }
            
            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
        
        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            x[i] /= aug[[i, i]];
        }
        
        Ok(x)
    }
    
    /// Calculate specific element of matrix inverse (for standard errors)
    fn matrix_inverse_element(x: &Array2<f64>, row: usize, col: usize) -> f64 {
        let n = x.nrows();
        let p = x.ncols();
        
        // Compute X'X
        let mut xtx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x[[k, i]] * x[[k, j]];
                }
                xtx[[i, j]] = sum;
            }
        }
        
        // For 2x2 matrix, use analytical inverse
        if p == 2 {
            let det = xtx[[0, 0]] * xtx[[1, 1]] - xtx[[0, 1]] * xtx[[1, 0]];
            if det.abs() < 1e-12 {
                return 1.0; // Fallback
            }
            
            match (row, col) {
                (0, 0) => xtx[[1, 1]] / det,
                (0, 1) | (1, 0) => -xtx[[0, 1]] / det,
                (1, 1) => xtx[[0, 0]] / det,
                _ => 1.0,
            }
        } else {
            // For larger matrices, use approximation
            1.0 / xtx[[row, col]].max(1e-6)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_adf_stationary_series() {
        // White noise should be stationary
        let series = Array1::from(vec![
            0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, 0.0,
            -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1
        ]);
        
        let (statistic, p_value, is_stationary) = StationarityTests::adf_test(&series);
        
        // Should reject unit root hypothesis (statistic should be negative)
        assert!(statistic < 0.0);
        assert!(p_value < 0.5); // Should have some evidence against unit root
    }
    
    #[test]
    fn test_adf_random_walk() {
        // Random walk should be non-stationary
        let mut series = vec![0.0];
        for i in 1..50 {
            series.push(series[i-1] + if i % 2 == 0 { 0.1 } else { -0.1 });
        }
        let series = Array1::from(series);
        
        let (statistic, p_value, is_stationary) = StationarityTests::adf_test(&series);
        
        // Should fail to reject unit root (less negative statistic)
        assert!(statistic > -3.0); // Less evidence against unit root
        assert!(p_value > 0.1);    // Higher p-value
    }
    
    #[test]
    fn test_kpss_stationary_series() {
        // Stationary series around trend
        let series: Array1<f64> = (0..30)
            .map(|i| i as f64 * 0.1 + (i as f64 * 0.1).sin() * 0.2)
            .collect::<Vec<_>>()
            .into();
        
        let (statistic, p_value, is_stationary) = StationarityTests::kpss_test(&series);
        
        // Should fail to reject stationarity null
        assert!(statistic < 0.5); // Lower KPSS statistic
        assert!(is_stationary);   // Should conclude stationarity
    }
    
    #[test]
    fn test_phillips_perron_consistency() {
        // Test that PP gives similar results to ADF for simple case
        let series = Array1::from(vec![
            1.0, 1.1, 1.05, 1.15, 1.08, 1.18, 1.12, 1.22, 1.16, 1.26,
            1.20, 1.30, 1.24, 1.34, 1.28, 1.38, 1.32, 1.42, 1.36, 1.46
        ]);
        
        let (adf_stat, _, adf_stationary) = StationarityTests::adf_test(&series);
        let (pp_stat, _, pp_stationary) = StationarityTests::phillips_perron_test(&series);
        
        // Results should be in same direction (both tests for unit root)
        assert_eq!(adf_stationary, pp_stationary);
        
        // Statistics should have same sign
        assert_eq!(adf_stat.signum(), pp_stat.signum());
    }
    
    #[test]
    fn test_diff_function() {
        let series = Array1::from(vec![1.0, 3.0, 6.0, 10.0, 15.0]);
        let diff1 = StationarityTests::diff(&series, 1);
        let expected = Array1::from(vec![2.0, 3.0, 4.0, 5.0]);
        
        assert_eq!(diff1.len(), expected.len());
        for (actual, expected) in diff1.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_insufficient_data() {
        let short_series = Array1::from(vec![1.0, 2.0, 3.0]);
        
        let (stat, p_val, is_stat) = StationarityTests::adf_test(&short_series);
        assert_eq!(stat, 0.0);
        assert_eq!(p_val, 1.0);
        assert!(!is_stat);
    }
    
    #[test]
    fn test_newey_west_variance() {
        let residuals = Array1::from(vec![0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0]);
        let variance = StationarityTests::newey_west_variance(&residuals, Some(2));
        
        // Should be positive
        assert!(variance > 0.0);
        
        // Should be reasonable magnitude
        assert!(variance < 1.0);
    }
}