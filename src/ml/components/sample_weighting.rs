//! Sample weighting strategies for ML models
//!
//! This module provides shared sample weighting functionality extracted from both
//! pattern_model_example.rs and classifier_model_example.rs. It implements volatility-based
//! and pattern-based weighting strategies using the extract_safe! macro for safety.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use crate::extract_safe;

/// Volatility-based sample weighting strategy
///
/// This struct implements volatility-based sample weighting as used in
/// ScientificTradingClassifier, where samples are weighted based on their
/// return magnitude relative to recent volatility.
#[derive(Debug, Clone)]
pub struct VolatilityWeighting {
    /// Rolling window size for volatility calculation
    pub window_size: usize,
    /// Minimum weight clamp value
    pub min_weight: f32,
    /// Maximum weight clamp value
    pub max_weight: f32,
}

impl VolatilityWeighting {
    /// Create a new volatility weighting strategy
    ///
    /// # Parameters
    /// - `window_size`: Rolling window for volatility calculation
    /// - `min_weight`: Minimum weight value (default: 0.1)
    /// - `max_weight`: Maximum weight value (default: 3.0)
    pub fn new(window_size: usize, min_weight: f32, max_weight: f32) -> Self {
        Self {
            window_size,
            min_weight,
            max_weight,
        }
    }

    /// Create default volatility weighting (20-period window, 0.1-3.0 range)
    pub fn default() -> Self {
        Self::new(20, 0.1, 3.0)
    }

    /// Calculate volatility-based sample weights
    ///
    /// # Parameters
    /// - `py`: Python context
    /// - `returns`: Array of return values
    ///
    /// # Returns
    /// PyArray1<f32> containing sample weights
    ///
    /// # Algorithm
    /// For each sample i:
    /// 1. Calculate rolling average absolute return over window
    /// 2. Weight = (current_abs_return / avg_abs_return).clamp(min_weight, max_weight)
    pub fn calculate_weights<'py>(
        &self,
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let returns_array = returns.as_array();
        let rets = extract_safe!(returns_array, "returns");
        let n = rets.len();
        let mut weights = vec![1.0f32; n];

        let window = self.window_size.min(n);
        
        for i in window..n {
            let window_start = i.saturating_sub(window);
            let window_rets = &rets[window_start..i];
            let abs_ret = rets[i].abs();
            let avg_abs_ret = window_rets.iter()
                .map(|r| r.abs())
                .sum::<f32>() / window_rets.len() as f32;

            if avg_abs_ret > 0.0 {
                weights[i] = (abs_ret / avg_abs_ret).clamp(self.min_weight, self.max_weight);
            }
        }

        Ok(PyArray1::from_vec(py, weights).to_owned().into())
    }
}

/// Pattern-based sample weighting strategy
///
/// This struct implements pattern-based sample weighting as used in
/// PatternRecognitionClassifier, where samples are weighted based on
/// pattern rarity and market volatility.
#[derive(Debug, Clone)]
pub struct PatternWeighting {
    /// Volatility normalization target (e.g., 0.02 for 2%)
    pub volatility_target: f32,
    /// Minimum weight clamp value
    pub min_weight: f32,
    /// Maximum weight clamp value
    pub max_weight: f32,
}

impl PatternWeighting {
    /// Create a new pattern weighting strategy
    ///
    /// # Parameters
    /// - `volatility_target`: Target volatility for normalization
    /// - `min_weight`: Minimum weight value
    /// - `max_weight`: Maximum weight value
    pub fn new(volatility_target: f32, min_weight: f32, max_weight: f32) -> Self {
        Self {
            volatility_target,
            min_weight,
            max_weight,
        }
    }

    /// Create default pattern weighting (2% vol target, 0.5-2.0 range)
    pub fn default() -> Self {
        Self::new(0.02, 0.5, 2.0)
    }

    /// Calculate pattern-based sample weights
    ///
    /// # Parameters
    /// - `py`: Python context
    /// - `pattern_signals`: 2D array of pattern signals (samples × patterns)
    /// - `volatility`: Array of volatility estimates
    ///
    /// # Returns
    /// PyArray1<f32> containing sample weights
    ///
    /// # Algorithm
    /// For each sample i:
    /// 1. Count active patterns (signal > 0.5)
    /// 2. Calculate rarity weight = 1 / sqrt(pattern_count) if patterns exist
    /// 3. Calculate volatility weight = (vol[i] / target_vol).clamp(min, max)
    /// 4. Final weight = rarity_weight * vol_weight
    pub fn calculate_pattern_weights<'py>(
        &self,
        py: Python<'py>,
        pattern_signals: PyReadonlyArray2<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let signals = pattern_signals.as_array();
        let volatility_array = volatility.as_array();
        let vols = extract_safe!(volatility_array, "volatility");
        let n = signals.nrows();

        if vols.len() != n {
            return Err(PyValueError::new_err("Pattern signals and volatility arrays must have same length"));
        }

        let mut weights = vec![1.0f32; n];

        for i in 0..n {
            let pattern_row = signals.row(i);
            let pattern_count = pattern_row.iter().filter(|&&x| x > 0.5).count() as f32;
            let vol_weight = (vols[i] / self.volatility_target).clamp(self.min_weight, self.max_weight);

            // Weight by pattern rarity and market volatility
            let rarity_weight = if pattern_count > 0.0 { 
                1.0 / pattern_count.sqrt() 
            } else { 
                1.0 
            };
            
            weights[i] = rarity_weight * vol_weight;
        }

        Ok(PyArray1::from_vec(py, weights).to_owned().into())
    }
}

/// Unified sample weight calculator
///
/// This struct provides a unified interface for calculating sample weights
/// using different strategies, implementing the LabelGenerator trait.
#[derive(Debug, Clone)]
pub struct SampleWeightCalculator {
    /// Volatility weighting strategy
    pub volatility_weighting: VolatilityWeighting,
    /// Pattern weighting strategy
    pub pattern_weighting: PatternWeighting,
}

impl SampleWeightCalculator {
    /// Create a new sample weight calculator
    pub fn new(
        volatility_weighting: VolatilityWeighting,
        pattern_weighting: PatternWeighting,
    ) -> Self {
        Self {
            volatility_weighting,
            pattern_weighting,
        }
    }

    /// Create default sample weight calculator
    pub fn default() -> Self {
        Self::new(
            VolatilityWeighting::default(),
            PatternWeighting::default(),
        )
    }

    /// Calculate volatility-based weights
    pub fn calculate_volatility_weights<'py>(
        &self,
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.volatility_weighting.calculate_weights(py, returns)
    }

    /// Calculate pattern-based weights
    pub fn calculate_pattern_weights<'py>(
        &self,
        py: Python<'py>,
        pattern_signals: PyReadonlyArray2<'py, f32>,
        volatility: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.pattern_weighting.calculate_pattern_weights(py, pattern_signals, volatility)
    }

    /// Calculate combined weights using both volatility and pattern information
    ///
    /// # Parameters
    /// - `py`: Python context
    /// - `returns`: Array of return values
    /// - `pattern_signals`: Optional 2D array of pattern signals
    /// - `volatility`: Array of volatility estimates
    ///
    /// # Returns
    /// Combined sample weights
    pub fn calculate_combined_weights<'py>(
        &self,
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
        pattern_signals: Option<PyReadonlyArray2<'py, f32>>,
        volatility: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let vol_weights = self.calculate_volatility_weights(py, returns)?;
        
        if let Some(patterns) = pattern_signals {
            // For now, just return pattern weights when both are available
            // TODO: Implement proper weight combination
            self.calculate_pattern_weights(py, patterns, volatility)
        } else {
            Ok(vol_weights)
        }
    }
}

// Implement LabelGenerator trait for SampleWeightCalculator
impl crate::ml::traits::LabelGenerator for SampleWeightCalculator {
    fn create_triple_barrier_labels<'py>(
        &self,
        _py: Python<'py>,
        _prices: PyReadonlyArray1<'py, f32>,
        _volatility: PyReadonlyArray1<'py, f32>,
        _profit_mult: f32,
        _stop_mult: f32,
        _max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        Err(PyValueError::new_err("SampleWeightCalculator does not implement label generation"))
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
        Err(PyValueError::new_err("SampleWeightCalculator does not implement label generation"))
    }

    fn calculate_sample_weights<'py>(
        &self,
        py: Python<'py>,
        returns: PyReadonlyArray1<'py, f32>,
        volatility: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        if let Some(_vol) = volatility {
            self.calculate_volatility_weights(py, returns)
        } else {
            self.calculate_volatility_weights(py, returns)
        }
    }
}

// Ensure thread safety
unsafe impl Send for VolatilityWeighting {}
unsafe impl Sync for VolatilityWeighting {}
unsafe impl Send for PatternWeighting {}
unsafe impl Sync for PatternWeighting {}
unsafe impl Send for SampleWeightCalculator {}
unsafe impl Sync for SampleWeightCalculator {}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray2;

    #[test]
    fn test_volatility_weighting_creation() {
        let weighting = VolatilityWeighting::new(20, 0.1, 3.0);
        assert_eq!(weighting.window_size, 20);
        assert_eq!(weighting.min_weight, 0.1);
        assert_eq!(weighting.max_weight, 3.0);
    }

    #[test]
    fn test_pattern_weighting_creation() {
        let weighting = PatternWeighting::new(0.02, 0.5, 2.0);
        assert_eq!(weighting.volatility_target, 0.02);
        assert_eq!(weighting.min_weight, 0.5);
        assert_eq!(weighting.max_weight, 2.0);
    }

    #[test]
    fn test_sample_weight_calculator_creation() {
        let calculator = SampleWeightCalculator::default();
        assert_eq!(calculator.volatility_weighting.window_size, 20);
        assert_eq!(calculator.pattern_weighting.volatility_target, 0.02);
    }
}