//! Label generation components for ML models
//!
//! This module provides shared label generation functionality extracted from both
//! pattern_model_example.rs and classifier_model_example.rs. It implements triple barrier
//! method and pattern-based labeling using the extract_safe! macro for safety.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::extract_safe;
use crate::ml::traits::LabelGenerator;

/// Triple barrier label generator
///
/// This struct implements the triple barrier method for generating trading labels
/// as used in ScientificTradingClassifier. It considers profit targets, stop losses,
/// and maximum holding periods to determine optimal trading signals.
#[derive(Debug, Clone)]
pub struct TripleBarrierLabeler {
    /// Default profit multiplier
    pub default_profit_mult: f32,
    /// Default stop multiplier
    pub default_stop_mult: f32,
    /// Default maximum holding period
    pub default_max_hold: usize,
}

impl TripleBarrierLabeler {
    /// Create a new triple barrier labeler
    ///
    /// # Parameters
    /// - `profit_mult`: Default profit target multiplier
    /// - `stop_mult`: Default stop loss multiplier
    /// - `max_hold`: Default maximum holding period
    pub fn new(profit_mult: f32, stop_mult: f32, max_hold: usize) -> Self {
        Self {
            default_profit_mult: profit_mult,
            default_stop_mult: stop_mult,
            default_max_hold: max_hold,
        }
    }

    /// Create default triple barrier labeler
    pub fn default() -> Self {
        Self::new(2.0, 1.5, 20)
    }

    /// Generate triple barrier labels with custom parameters
    ///
    /// # Algorithm
    /// For each bar i:
    /// 1. Calculate profit_target = price[i] * (1 + profit_mult * volatility[i])
    /// 2. Calculate stop_target = price[i] * (1 - stop_mult * volatility[i])
    /// 3. Look ahead up to max_hold periods
    /// 4. Return 2 (buy) if profit target hit first
    /// 5. Return 0 (sell) if stop target hit first
    /// 6. Return 1 (hold) if neither hit within max_hold
    pub fn generate_labels<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
        profit_mult: Option<f32>,
        stop_mult: Option<f32>,
        max_hold: Option<usize>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let prices_array = prices.as_array();
        let volatility_array = volatility.as_array();
        let prices_slice = extract_safe!(prices_array, "prices");
        let vols_slice = extract_safe!(volatility_array, "volatility");

        if prices_slice.len() != vols_slice.len() {
            return Err(PyValueError::new_err("Price and volatility arrays must have same length"));
        }

        let n = prices_slice.len();
        let mut labels = vec![1i32; n];

        let profit_mult = profit_mult.unwrap_or(self.default_profit_mult);
        let stop_mult = stop_mult.unwrap_or(self.default_stop_mult);
        let max_hold = max_hold.unwrap_or(self.default_max_hold);

        for i in 0..n.saturating_sub(max_hold) {
            let entry = prices_slice[i];
            let vol = vols_slice[i];

            if vol <= 0.0 || entry <= 0.0 {
                continue;
            }

            let profit_target = entry * (1.0 + profit_mult * vol);
            let stop_target = entry * (1.0 - stop_mult * vol);

            for j in (i + 1)..=(i + max_hold).min(n - 1) {
                let price = prices_slice[j];

                if price >= profit_target {
                    labels[i] = 2; // Buy signal
                    break;
                } else if price <= stop_target {
                    labels[i] = 0; // Sell signal
                    break;
                } else if j == (i + max_hold).min(n - 1) {
                    // Time-based exit
                    let ret = (price / entry) - 1.0;
                    labels[i] = if ret > 0.002 { 2 } else if ret < -0.002 { 0 } else { 1 };
                }
            }
        }

        Ok(PyArray1::from_vec(py, labels).to_owned().into())
    }
}

impl LabelGenerator for TripleBarrierLabeler {
    fn create_triple_barrier_labels<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
        profit_mult: f32,
        stop_mult: f32,
        max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.generate_labels(py, prices, volatility, Some(profit_mult), Some(stop_mult), Some(max_hold))
    }

    fn create_pattern_labels<'py>(
        &self,
        _py: Python<'py>,
        _open_prices: PyReadonlyArray1<'py, f32>,
        _high_prices: PyReadonlyArray1<'py, f32>,
        _low_prices: PyReadonlyArray1<'py, f32>,
        _close_prices: PyReadonlyArray1<'py, f32>,
        _future_periods: usize,
        _profit_threshold: f32,
        _stop_threshold: f32,
    ) -> PyResult<Py<PyArray1<i32>>> {
        Err(PyValueError::new_err("TripleBarrierLabeler does not implement pattern labels"))
    }

    fn calculate_sample_weights<'py>(
        &self,
        _py: Python<'py>,
        _returns: PyReadonlyArray1<'py, f32>,
        _volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        Err(PyValueError::new_err("TripleBarrierLabeler does not implement sample weighting"))
    }
}

/// Pattern-based label generator
///
/// This struct implements pattern-based label generation as used in
/// PatternRecognitionClassifier, combining pattern signals with future price movements.
#[derive(Debug, Clone)]
pub struct PatternLabeler {
    /// Default future periods to look ahead
    pub default_future_periods: usize,
    /// Default profit threshold
    pub default_profit_threshold: f32,
    /// Default stop threshold
    pub default_stop_threshold: f32,
}

impl PatternLabeler {
    /// Create a new pattern labeler
    ///
    /// # Parameters
    /// - `future_periods`: Default periods to look ahead
    /// - `profit_threshold`: Default profit threshold for buy signal
    /// - `stop_threshold`: Default stop threshold for sell signal
    pub fn new(future_periods: usize, profit_threshold: f32, stop_threshold: f32) -> Self {
        Self {
            default_future_periods: future_periods,
            default_profit_threshold: profit_threshold,
            default_stop_threshold: stop_threshold,
        }
    }

    /// Create default pattern labeler
    pub fn default() -> Self {
        Self::new(10, 0.02, 0.02)
    }

    /// Generate pattern-based labels
    ///
    /// # Algorithm
    /// For each bar i:
    /// 1. Look ahead up to future_periods
    /// 2. Check if high/low breaches profit/stop thresholds
    /// 3. Return first signal hit, or final return-based signal
    pub fn generate_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        open_prices: PyReadonlyArray1<'py, f32>,
        high_prices: PyReadonlyArray1<'py, f32>,
        low_prices: PyReadonlyArray1<'py, f32>,
        close_prices: PyReadonlyArray1<'py, f32>,
        future_periods: Option<usize>,
        profit_threshold: Option<f32>,
        stop_threshold: Option<f32>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let open_array = open_prices.as_array();
        let high_array = high_prices.as_array();
        let low_array = low_prices.as_array();
        let close_array = close_prices.as_array();
        let opens = extract_safe!(open_array, "open_prices");
        let highs = extract_safe!(high_array, "high_prices");
        let lows = extract_safe!(low_array, "low_prices");
        let closes = extract_safe!(close_array, "close_prices");

        let n = opens.len();
        if highs.len() != n || lows.len() != n || closes.len() != n {
            return Err(PyValueError::new_err("All OHLC arrays must have same length"));
        }

        let mut labels = vec![1i32; n];
        let future_periods = future_periods.unwrap_or(self.default_future_periods);
        let profit_threshold = profit_threshold.unwrap_or(self.default_profit_threshold);
        let stop_threshold = stop_threshold.unwrap_or(self.default_stop_threshold);

        for i in 0..(n.saturating_sub(future_periods)) {
            let entry_price = closes[i];
            let mut signal = 1; // Default hold

            for j in (i + 1)..=(i + future_periods).min(n - 1) {
                let current_high = highs[j];
                let current_low = lows[j];
                let return_up = (current_high / entry_price) - 1.0;
                let return_down = (current_low / entry_price) - 1.0;

                if return_up >= profit_threshold {
                    signal = 2; // Buy signal
                    break;
                } else if return_down <= -stop_threshold {
                    signal = 0; // Sell signal
                    break;
                }
            }

            // If no clear signal, use final return
            if signal == 1 {
                let final_price = closes[(i + future_periods).min(n - 1)];
                let final_return = (final_price / entry_price) - 1.0;
                signal = if final_return > 0.005 { 2 } else if final_return < -0.005 { 0 } else { 1 };
            }

            labels[i] = signal;
        }

        Ok(PyArray1::from_vec(py, labels).to_owned().into())
    }
}

impl LabelGenerator for PatternLabeler {
    fn create_triple_barrier_labels<'py>(
        &self,
        _py: Python<'py>,
        _prices: PyReadonlyArray1<'py, f32>,
        _volatility: PyReadonlyArray1<'py, f32>,
        _profit_mult: f32,
        _stop_mult: f32,
        _max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        Err(PyValueError::new_err("PatternLabeler does not implement triple barrier labels"))
    }

    fn create_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        open_prices: PyReadonlyArray1<'py, f32>,
        high_prices: PyReadonlyArray1<'py, f32>,
        low_prices: PyReadonlyArray1<'py, f32>,
        close_prices: PyReadonlyArray1<'py, f32>,
        future_periods: usize,
        profit_threshold: f32,
        stop_threshold: f32,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.generate_pattern_labels(
            py,
            open_prices,
            high_prices,
            low_prices,
            close_prices,
            Some(future_periods),
            Some(profit_threshold),
            Some(stop_threshold),
        )
    }

    fn calculate_sample_weights<'py>(
        &self,
        _py: Python<'py>,
        _returns: PyReadonlyArray1<'py, f32>,
        _volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        Err(PyValueError::new_err("PatternLabeler does not implement sample weighting"))
    }
}

/// Unified label generator combining both strategies
///
/// This struct provides a unified interface for generating labels using
/// different strategies, implementing the full LabelGenerator trait.
#[derive(Debug, Clone)]
pub struct ComponentLabelGenerator {
    /// Triple barrier labeler
    pub triple_barrier: TripleBarrierLabeler,
    /// Pattern labeler
    pub pattern_labeler: PatternLabeler,
}

impl ComponentLabelGenerator {
    /// Create a new unified label generator
    pub fn new(triple_barrier: TripleBarrierLabeler, pattern_labeler: PatternLabeler) -> Self {
        Self {
            triple_barrier,
            pattern_labeler,
        }
    }

    /// Create default unified label generator
    pub fn default() -> Self {
        Self::new(
            TripleBarrierLabeler::default(),
            PatternLabeler::default(),
        )
    }
}

impl LabelGenerator for ComponentLabelGenerator {
    fn create_triple_barrier_labels<'py>(
        &self,
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
        profit_mult: f32,
        stop_mult: f32,
        max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.triple_barrier.create_triple_barrier_labels(
            py, prices, volatility, profit_mult, stop_mult, max_hold
        )
    }

    fn create_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        open_prices: PyReadonlyArray1<'py, f32>,
        high_prices: PyReadonlyArray1<'py, f32>,
        low_prices: PyReadonlyArray1<'py, f32>,
        close_prices: PyReadonlyArray1<'py, f32>,
        future_periods: usize,
        profit_threshold: f32,
        stop_threshold: f32,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.pattern_labeler.create_pattern_labels(
            py, open_prices, high_prices, low_prices, close_prices,
            future_periods, profit_threshold, stop_threshold
        )
    }

    fn calculate_sample_weights<'py>(
        &self,
        _py: Python<'py>,
        _returns: PyReadonlyArray1<'py, f32>,
        _volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        Err(PyValueError::new_err("ComponentLabelGenerator does not implement sample weighting - use SampleWeightCalculator"))
    }
}

// Ensure thread safety
unsafe impl Send for TripleBarrierLabeler {}
unsafe impl Sync for TripleBarrierLabeler {}
unsafe impl Send for PatternLabeler {}
unsafe impl Sync for PatternLabeler {}
unsafe impl Send for ComponentLabelGenerator {}
unsafe impl Sync for ComponentLabelGenerator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_barrier_creation() {
        let labeler = TripleBarrierLabeler::new(2.0, 1.5, 20);
        assert_eq!(labeler.default_profit_mult, 2.0);
        assert_eq!(labeler.default_stop_mult, 1.5);
        assert_eq!(labeler.default_max_hold, 20);
    }

    #[test]
    fn test_pattern_labeler_creation() {
        let labeler = PatternLabeler::new(10, 0.02, 0.02);
        assert_eq!(labeler.default_future_periods, 10);
        assert_eq!(labeler.default_profit_threshold, 0.02);
        assert_eq!(labeler.default_stop_threshold, 0.02);
    }

    #[test]
    fn test_unified_label_generator_creation() {
        let generator = ComponentLabelGenerator::default();
        assert_eq!(generator.triple_barrier.default_profit_mult, 2.0);
        assert_eq!(generator.pattern_labeler.default_future_periods, 10);
    }
}