use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::indicators::api::{BollingerBandsOutput, HilbertTransformOutput};
use std::f64::consts::PI;

pub fn rsi_cpu<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut results = vec![f64::NAN; len];
    if len < period + 1 {
        return Ok(PyArray1::from_vec(py, results).to_owned().into());
    }
    let mut gains = Vec::with_capacity(len.saturating_sub(1));
    let mut losses = Vec::with_capacity(len.saturating_sub(1));
    for i in 1..len {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;
    for i in period..len {
        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            results[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            results[i] = 100.0;
        }
        if i < gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn ema_cpu<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut results = vec![f64::NAN; len];
    if len == 0 {
        return Ok(PyArray1::from_vec(py, results).to_owned().into());
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema_value = prices[0];
    results[0] = ema_value;
    for i in 1..len {
        ema_value = alpha * prices[i] + (1.0 - alpha) * ema_value;
        results[i] = ema_value;
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn sma_cpu<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let values = values.as_array();
    let len = values.len();
    let mut results = vec![f64::NAN; len];
    if period == 0 {
        return Ok(PyArray1::from_vec(py, results).to_owned().into());
    }
    for (i, item) in results.iter_mut().enumerate().take(len) {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let sum: f64 = (start_idx..=i).map(|j| values[j]).sum();
            *item = sum / period as f64;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn bollinger_bands_cpu<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<BollingerBandsOutput> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut upper_vec = vec![f64::NAN; len];
    let mut middle_vec = vec![f64::NAN; len];
    let mut lower_vec = vec![f64::NAN; len];
    if period == 0 {
        return Ok(BollingerBandsOutput {
            upper: PyArray1::from_vec(py, upper_vec).to_owned().into(),
            middle: PyArray1::from_vec(py, middle_vec).to_owned().into(),
            lower: PyArray1::from_vec(py, lower_vec).to_owned().into(),
        });
    }
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window: Vec<f64> = (start_idx..=i).map(|j| prices[j]).collect();
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();
            upper_vec[i] = mean + std_dev * std;
            middle_vec[i] = mean;
            lower_vec[i] = mean - std_dev * std;
        }
    }
    Ok(BollingerBandsOutput {
        upper: PyArray1::from_vec(py, upper_vec).to_owned().into(),
        middle: PyArray1::from_vec(py, middle_vec).to_owned().into(),
        lower: PyArray1::from_vec(py, lower_vec).to_owned().into(),
    })
}

pub fn atr_cpu<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    let mut true_ranges = Vec::with_capacity(len);
    for i in 0..len {
        let tr = if i == 0 {
            high[i] - low[i]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        true_ranges.push(tr);
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let atr_value: f64 =
                (start_idx..=i).map(|j| true_ranges[j]).sum::<f64>() / period as f64;
            results[i] = atr_value;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn williams_r_cpu<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window_high = (start_idx..=i)
                .map(|j| high[j])
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            let window_low = (start_idx..=i)
                .map(|j| low[j])
                .fold(f64::INFINITY, |a, b| a.min(b));
            if window_high != window_low {
                let williams = -100.0 * (window_high - close[i]) / (window_high - window_low);
                results[i] = williams;
            } else {
                results[i] = 0.0;
            }
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn cci_cpu<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    let typical_prices: Vec<f64> = (0..len)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window: Vec<f64> = (start_idx..=i).map(|j| typical_prices[j]).collect();
            let sma: f64 = window.iter().sum::<f64>() / period as f64;
            let mean_deviation: f64 =
                window.iter().map(|x| (x - sma).abs()).sum::<f64>() / period as f64;
            let cci = if mean_deviation != 0.0 {
                (typical_prices[i] - sma) / (0.015 * mean_deviation)
            } else {
                0.0
            };
            results[i] = cci;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn calculate_supersmoother_coeffs(period: f64) -> (f64, f64, f64) {
    let a1 = (-1.414 * PI / period).exp();
    let b1 = 2.0 * a1 * (1.414 * PI / period).cos();
    let c2 = b1;
    let c3 = -a1 * a1;
    let c1 = 1.0 - c2 - c3;
    (c1, c2, c3)
}

pub fn supersmoother_cpu<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_array();
    let len = data.len();
    let mut results = vec![0.0; len];

    if period < 2 {
        return Ok(PyArray1::from_vec(py, results).to_owned().into());
    }
    if len < 3 {
        return Ok(PyArray1::from_vec(py, results).to_owned().into());
    }

    let (c1, c2, c3) = calculate_supersmoother_coeffs(period as f64);

    // Start calculation at index 2 (need 2 previous values)
    let mut i = 2;
    // Unrolled loop for potential SIMD optimization
    while i + 3 < len {
        results[i] = c1 * (data[i] + data[i - 1]) / 2.0 + c2 * results[i - 1] + c3 * results[i - 2];
        results[i + 1] = c1 * (data[i + 1] + data[i]) / 2.0 + c2 * results[i] + c3 * results[i - 1];
        results[i + 2] =
            c1 * (data[i + 2] + data[i + 1]) / 2.0 + c2 * results[i + 1] + c3 * results[i];
        results[i + 3] =
            c1 * (data[i + 3] + data[i + 2]) / 2.0 + c2 * results[i + 2] + c3 * results[i + 1];
        i += 4;
    }
    for j in i..len {
        results[j] = c1 * (data[j] + data[j - 1]) / 2.0 + c2 * results[j - 1] + c3 * results[j - 2];
    }

    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn vpin_cpu_kernel(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    let len = buy_volumes.len().min(sell_volumes.len());
    let mut output = vec![0.0; len];

    for i in window..len {
        let buy_sum: f64 = buy_volumes[i - window + 1..=i].iter().sum();
        let sell_sum: f64 = sell_volumes[i - window + 1..=i].iter().sum();

        let diff = buy_sum - sell_sum;
        let total = buy_sum + sell_sum;
        let imbalance = diff.abs();

        if total > 1e-12 {
            output[i] = imbalance / total;
        } else {
            output[i] = 0.0;
        }
    }

    output
}

/// Calculate two-pole high-pass filter coefficients
pub fn calculate_highpass_coeffs(period: f64) -> (f64, f64, f64) {
    let alpha1 = (2.0 * PI / period).cos();
    let beta1 = (2.0 * PI / period).sin();
    let gamma1 = 1.0 / (alpha1 + beta1);
    let alpha2 = gamma1 * alpha1;
    let beta2 = gamma1 * beta1;
    (alpha2, beta2, gamma1)
}

/// Apply two-pole high-pass filter (Ehlers roofing filter step 1)
pub fn roofing_highpass_cpu(data: &[f64], period: f64) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![0.0; len];

    if len < 3 {
        return result;
    }

    let (alpha, beta, _gamma) = calculate_highpass_coeffs(period);

    // Initialize first two values
    result[0] = 0.0;
    result[1] = 0.0;

    // Apply high-pass filter starting from index 2
    for i in 2..len {
        result[i] = (1.0 - alpha / 2.0) * (data[i] - data[i - 1]) + (1.0 - alpha) * result[i - 1]
            - (alpha - beta) * result[i - 2];
    }

    result
}

/// Apply roofing filter: high-pass followed by SuperSmoother low-pass
pub fn roofing_filter_cpu(data: &[f64], hp_period: f64, lp_period: usize) -> Vec<f64> {
    // Step 1: Apply high-pass filter
    let highpassed = roofing_highpass_cpu(data, hp_period);

    // Step 2: Apply SuperSmoother low-pass filter
    let len = highpassed.len();
    let mut result = vec![0.0; len];

    if lp_period < 2 || len < 3 {
        return result;
    }

    let (c1, c2, c3) = calculate_supersmoother_coeffs(lp_period as f64);

    // Apply SuperSmoother starting from index 2
    for i in 2..len {
        result[i] = c1 * (highpassed[i] + highpassed[i - 1]) / 2.0
            + c2 * result[i - 1]
            + c3 * result[i - 2];
    }

    result
}

/// Apply Automatic Gain Control (AGC) normalization
pub fn apply_agc_normalization_cpu(data: &[f64], decay_factor: f64) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![0.0; len];
    let mut peak = 0.0;

    for i in 0..len {
        // Update peak with decay
        peak = decay_factor * peak + (1.0 - decay_factor) * data[i].abs();

        // Normalize by peak, avoiding division by zero
        if peak > 1e-10 {
            result[i] = data[i] / peak;
        } else {
            result[i] = 0.0;
        }
    }

    result
}

/// Calculate quadrature component using one-bar difference method
pub fn calculate_quadrature_cpu(real_component: &[f64]) -> Vec<f64> {
    let len = real_component.len();
    let mut quadrature = vec![0.0; len];

    // One-bar difference: Q[i] = Real[i] - Real[i-1]
    for i in 1..len {
        quadrature[i] = real_component[i] - real_component[i - 1];
    }

    quadrature
}

/// Apply SuperSmoother filter to data with given period
pub fn apply_supersmoother(data: &[f64], period: f64) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![0.0; len];

    if period < 2.0 || len < 3 {
        return result;
    }

    let (c1, c2, c3) = calculate_supersmoother_coeffs(period);

    // Apply SuperSmoother starting from index 2
    for i in 2..len {
        result[i] = c1 * (data[i] + data[i - 1]) / 2.0 + c2 * result[i - 1] + c3 * result[i - 2];
    }

    result
}

/// Core Hilbert Transform implementation - single source of truth
pub fn hilbert_transform_core(data: &[f64], lp_period: usize) -> (Vec<f64>, Vec<f64>) {
    let len = data.len();

    // Initialize output arrays
    let mut real_component = vec![0.0; len];
    let imaginary_component;

    // Early return for insufficient data
    if len < 50 {
        // Need at least 48 + a few extra for the algorithm
        return (real_component, vec![0.0; len]);
    }

    // Step 1: Apply roofing filter (48-period high-pass + lp_period low-pass)
    let roofed_data = roofing_filter_cpu(data, 48.0, lp_period);

    // Step 2: Apply AGC normalization to get real component
    real_component = apply_agc_normalization_cpu(&roofed_data, 0.991);

    // Step 3: Calculate quadrature using one-bar difference
    let quadrature = calculate_quadrature_cpu(&real_component);

    // Step 4: Apply AGC to quadrature
    let agc_quadrature = apply_agc_normalization_cpu(&quadrature, 0.991);

    // Step 5: Apply SuperSmoother to AGC'd quadrature (imaginary component)
    imaginary_component = apply_supersmoother(&agc_quadrature, 10.0);

    (real_component, imaginary_component)
}

/// Main Hilbert Transform CPU implementation
pub fn hilbert_transform_cpu<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lp_period: usize,
) -> PyResult<HilbertTransformOutput> {
    let data_array = data.as_array();
    let data_slice = data_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "Failed to extract slice from data: array is not contiguous"
        )
    })?;

    let (real_component, imaginary_component) = hilbert_transform_core(data_slice, lp_period);

    Ok(HilbertTransformOutput {
        real: PyArray1::from_vec(py, real_component).to_owned().into(),
        imag: PyArray1::from_vec(py, imaginary_component).to_owned().into(),
    })
}
