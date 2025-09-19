//! Label generation components for ML models
//!
//! This module provides shared label generation functionality extracted from both
//! pattern_model_example.rs and classifier_model_example.rs. It implements triple barrier
//! method and pattern-based labeling using the extract_safe! macro for safety.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};
use rayon::prelude::*;
use crate::extract_safe;
use crate::ml::traits::LabelGenerator;

/// Trading side for side-aware labeling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSide {
    Long,
    Short,
}

/// Result of barrier touching with timing information
#[derive(Debug, Clone, Copy)]
struct BarrierResult {
    barrier_type: BarrierType,
    touch_time: usize,
}

/// Type of barrier touched first
#[derive(Debug, Clone, Copy, PartialEq)]
enum BarrierType {
    Upper,
    Lower,
    Vertical,
    None,
}

/// Triple barrier label generator with side-aware and path-dependent logic
///
/// This struct implements the López de Prado triple barrier method with:
/// - Side-aware labeling (8 configurations)
/// - Path-dependent barrier logic
/// - Parallel processing with rayon
#[derive(Debug, Clone)]
pub struct TripleBarrierLabeler {
    /// Default profit multiplier
    pub default_profit_mult: f32,
    /// Default stop multiplier
    pub default_stop_mult: f32,
    /// Default maximum holding period
    pub default_max_hold: usize,
    /// Trading side for side-aware labeling
    pub trading_side: Option<TradingSide>,
}

impl TripleBarrierLabeler {
    /// Create a new triple barrier labeler
    pub fn new(profit_mult: f32, stop_mult: f32, max_hold: usize) -> Self {
        Self {
            default_profit_mult: profit_mult,
            default_stop_mult: stop_mult,
            default_max_hold: max_hold,
            trading_side: None,
        }
    }

    /// Create default triple barrier labeler
    pub fn default() -> Self {
        Self::new(2.0, 1.5, 20)
    }
}

impl TripleBarrierLabeler {
    /// Set trading side for side-aware labeling
    fn with_side(mut self, side: TradingSide) -> Self {
        self.trading_side = Some(side);
        self
    }

    /// Path-dependent barrier checking
    fn check_barriers_path_dependent(
        &self,
        prices: &[f32],
        start_idx: usize,
        _entry_price: f32,
        upper_barrier: f32,
        lower_barrier: f32,
        max_hold: usize,
    ) -> BarrierResult {
        let end_idx = (start_idx + max_hold).min(prices.len() - 1);
        
        for (i, &price) in prices.iter().enumerate().take(end_idx + 1).skip(start_idx + 1) {
            // Check barriers in chronological order
            if price >= upper_barrier {
                return BarrierResult {
                    barrier_type: BarrierType::Upper,
                    touch_time: i - start_idx,
                };
            }
            
            if price <= lower_barrier {
                return BarrierResult {
                    barrier_type: BarrierType::Lower,
                    touch_time: i - start_idx,
                };
            }
        }
        
        // If we reach max_hold without hitting barriers
        if end_idx == start_idx + max_hold {
            BarrierResult {
                barrier_type: BarrierType::Vertical,
                touch_time: max_hold,
            }
        } else {
            BarrierResult {
                barrier_type: BarrierType::None,
                touch_time: end_idx - start_idx,
            }
        }
    }

    /// Generate side-aware label based on barrier result
    fn generate_side_aware_label(
        &self,
        barrier_result: BarrierResult,
        entry_price: f32,
        exit_price: f32,
        side: Option<TradingSide>,
    ) -> i32 {
        match (barrier_result.barrier_type, side) {
            // Long position configurations
            (BarrierType::Upper, Some(TradingSide::Long)) => 1,  // Profit hit
            (BarrierType::Lower, Some(TradingSide::Long)) => -1, // Loss hit
            (BarrierType::Vertical, Some(TradingSide::Long)) => {
                let return_val = (exit_price / entry_price) - 1.0;
                if return_val > 0.002 { 1 } else if return_val < -0.002 { -1 } else { 0 }
            },
            (BarrierType::None, Some(TradingSide::Long)) => 0, // NaN equivalent
            
            // Short position configurations
            (BarrierType::Upper, Some(TradingSide::Short)) => -1, // Loss hit (price went up)
            (BarrierType::Lower, Some(TradingSide::Short)) => 1,  // Profit hit (price went down)
            (BarrierType::Vertical, Some(TradingSide::Short)) => {
                let return_val = (exit_price / entry_price) - 1.0;
                // For short positions, negative returns are profits
                if return_val < -0.002 { 1 } else if return_val > 0.002 { -1 } else { 0 }
            },
            (BarrierType::None, Some(TradingSide::Short)) => 0, // NaN equivalent
            
            // No side specified - traditional labeling
            (BarrierType::Upper, None) => 2,  // Buy signal
            (BarrierType::Lower, None) => 0,  // Sell signal
            (BarrierType::Vertical, None) => {
                let return_val = (exit_price / entry_price) - 1.0;
                if return_val > 0.002 { 2 } else if return_val < -0.002 { 0 } else { 1 }
            },
            (BarrierType::None, None) => 1, // Hold
        }
    }

    /// Generate triple barrier labels with parallel processing
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
        let profit_mult = profit_mult.unwrap_or(self.default_profit_mult);
        let stop_mult = stop_mult.unwrap_or(self.default_stop_mult);
        let max_hold = max_hold.unwrap_or(self.default_max_hold);

        // Parallel processing with rayon
        let labels: Vec<i32> = (0..n.saturating_sub(max_hold))
            .into_par_iter()
            .map(|i| {
                let entry = prices_slice[i];
                let vol = vols_slice[i];

                if vol <= 0.0 || entry <= 0.0 {
                    return 1; // Default hold
                }

                let upper_barrier = entry * (1.0 + profit_mult * vol);
                let lower_barrier = entry * (1.0 - stop_mult * vol);

                let barrier_result = self.check_barriers_path_dependent(
                    prices_slice,
                    i,
                    entry,
                    upper_barrier,
                    lower_barrier,
                    max_hold,
                );

                let exit_idx = (i + barrier_result.touch_time).min(n - 1);
                let exit_price = prices_slice[exit_idx];

                self.generate_side_aware_label(
                    barrier_result,
                    entry,
                    exit_price,
                    self.trading_side,
                )
            })
            .collect();

        // Pad remaining positions with default hold
        let mut full_labels = labels;
        full_labels.resize(n, 1);

        Ok(PyArray1::from_vec(py, full_labels).to_owned().into())
    }
}

impl Default for TripleBarrierLabeler {
    fn default() -> Self {
        Self::default()
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
        _ohlc_data: crate::ml::traits::OHLCData<'py>,
        _params: crate::ml::traits::PatternLabelingParams,
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
}

impl Default for PatternLabeler {
    fn default() -> Self {
        Self::default()
    }
}

impl PatternLabeler {
    /// Generate pattern-based labels
    fn generate_pattern_labels<'py>(
        &self,
        py: Python<'py>,
        ohlc_data: crate::ml::traits::OHLCData<'py>,
        params: crate::ml::traits::PatternLabelingParams,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let opens = ohlc_data.open_prices.as_array();
        let highs = ohlc_data.high_prices.as_array();
        let lows = ohlc_data.low_prices.as_array();
        let closes = ohlc_data.close_prices.as_array();

        let n = opens.len();
        if highs.len() != n || lows.len() != n || closes.len() != n {
            return Err(PyValueError::new_err("All OHLC arrays must have same length"));
        }

        let mut labels = vec![1i32; n];
        let future_periods = params.future_periods;
        let profit_threshold = params.profit_threshold;
        let stop_threshold = params.stop_threshold;

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
        ohlc_data: crate::ml::traits::OHLCData<'py>,
        params: crate::ml::traits::PatternLabelingParams,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.generate_pattern_labels(py, ohlc_data, params)
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

impl Default for ComponentLabelGenerator {
    fn default() -> Self {
        Self::default()
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
        ohlc_data: crate::ml::traits::OHLCData<'py>,
        params: crate::ml::traits::PatternLabelingParams,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.pattern_labeler.create_pattern_labels(py, ohlc_data, params)
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
        assert!(labeler.trading_side.is_none());
    }

    #[test]
    fn test_triple_barrier_with_side() {
        let labeler = TripleBarrierLabeler::new(2.0, 1.5, 20)
            .with_side(TradingSide::Long);
        assert_eq!(labeler.trading_side, Some(TradingSide::Long));
    }

    #[test]
    fn test_trading_side_enum() {
        assert_eq!(TradingSide::Long, TradingSide::Long);
        assert_ne!(TradingSide::Long, TradingSide::Short);
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