use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

pub fn rsi_cpu<'py>(py: Python<'py>, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut results = vec![f64::NAN; len];
    if len < period + 1 { return Ok(PyArray1::from_vec(py, results).to_owned().into()); }
    let mut gains = Vec::with_capacity(len.saturating_sub(1));
    let mut losses = Vec::with_capacity(len.saturating_sub(1));
    for i in 1..len {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 { gains.push(change); losses.push(0.0); } else { gains.push(0.0); losses.push(-change); }
    }
    let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;
    for i in period..len {
        if avg_loss != 0.0 { let rs = avg_gain / avg_loss; results[i] = 100.0 - (100.0 / (1.0 + rs)); } else { results[i] = 100.0; }
        if i < gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn ema_cpu<'py>(py: Python<'py>, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut results = vec![f64::NAN; len];
    if len == 0 { return Ok(PyArray1::from_vec(py, results).to_owned().into()); }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema_value = prices[0];
    results[0] = ema_value;
    for i in 1..len { ema_value = alpha * prices[i] + (1.0 - alpha) * ema_value; results[i] = ema_value; }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn sma_cpu<'py>(py: Python<'py>, values: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let values = values.as_array();
    let len = values.len();
    let mut results = vec![f64::NAN; len];
    if period == 0 { return Ok(PyArray1::from_vec(py, results).to_owned().into()); }
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let sum: f64 = (start_idx..=i).map(|j| values[j]).sum();
            results[i] = sum / period as f64;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn bollinger_bands_cpu<'py>(py: Python<'py>, prices: PyReadonlyArray1<f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let prices = prices.as_array();
    let len = prices.len();
    let mut upper = vec![f64::NAN; len];
    let mut middle = vec![f64::NAN; len];
    let mut lower = vec![f64::NAN; len];
    if period == 0 { return Ok((PyArray1::from_vec(py, upper).to_owned().into(), PyArray1::from_vec(py, middle).to_owned().into(), PyArray1::from_vec(py, lower).to_owned().into())); }
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window: Vec<f64> = (start_idx..=i).map(|j| prices[j]).collect();
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();
            upper[i] = mean + std_dev * std;
            middle[i] = mean;
            lower[i] = mean - std_dev * std;
        }
    }
    Ok((PyArray1::from_vec(py, upper).to_owned().into(), PyArray1::from_vec(py, middle).to_owned().into(), PyArray1::from_vec(py, lower).to_owned().into()))
}

pub fn atr_cpu<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    let mut true_ranges = Vec::with_capacity(len);
    for i in 0..len {
        let tr = if i == 0 { high[i] - low[i] } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        true_ranges.push(tr);
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let atr_value: f64 = (start_idx..=i).map(|j| true_ranges[j]).sum::<f64>() / period as f64;
            results[i] = atr_value;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}

pub fn williams_r_cpu<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window_high = (start_idx..=i).map(|j| high[j]).fold(f64::NEG_INFINITY, |a, b| a.max(b));
            let window_low = (start_idx..=i).map(|j| low[j]).fold(f64::INFINITY, |a, b| a.min(b));
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

pub fn cci_cpu<'py>(py: Python<'py>, high: PyReadonlyArray1<f64>, low: PyReadonlyArray1<f64>, close: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let len = high.len();
    let mut results = vec![f64::NAN; len];
    let typical_prices: Vec<f64> = (0..len).map(|i| (high[i] + low[i] + close[i]) / 3.0).collect();
    for i in 0..len {
        if i + 1 >= period {
            let start_idx = i + 1 - period;
            let window: Vec<f64> = (start_idx..=i).map(|j| typical_prices[j]).collect();
            let sma: f64 = window.iter().sum::<f64>() / period as f64;
            let mean_deviation: f64 = window.iter().map(|x| (x - sma).abs()).sum::<f64>() / period as f64;
            let cci = if mean_deviation != 0.0 { (typical_prices[i] - sma) / (0.015 * mean_deviation) } else { 0.0 };
            results[i] = cci;
        }
    }
    Ok(PyArray1::from_vec(py, results).to_owned().into())
}
