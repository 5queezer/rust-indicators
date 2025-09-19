//! Core financial time series implementation
//!
//! Provides the FinancialSeries struct with efficient operations for financial data analysis.

use ndarray::{s, Array1, ArrayView1};
use std::collections::BTreeMap;
use time::OffsetDateTime;

/// A financial time series with timestamps, values, and efficient indexing
///
/// This struct provides the foundation for all financial time series operations,
/// implementing zero-copy operations where possible and efficient algorithms
/// for common financial calculations.
///
/// # Fields
///
/// - `timestamps`: Ordered vector of timestamps
/// - `values`: Array of corresponding values  
/// - `index`: BTreeMap for O(log n) timestamp lookup
///
/// # Example
///
/// ```rust
/// use rust_indicators::financial::FinancialSeries;
/// use time::OffsetDateTime;
/// use ndarray::Array1;
///
/// let timestamps = vec![
///     OffsetDateTime::now_utc(),
///     OffsetDateTime::now_utc() + time::Duration::days(1),
/// ];
/// let values = Array1::from(vec![100.0, 102.0]);
/// let series = FinancialSeries::new(timestamps, values);
///
/// // Calculate percentage changes
/// let pct_changes = series.pct_change(1);
/// ```
#[derive(Debug, Clone)]
pub struct FinancialSeries {
    /// Ordered timestamps for the time series
    pub timestamps: Vec<OffsetDateTime>,
    /// Values corresponding to each timestamp
    pub values: Array1<f64>,
    /// Index mapping timestamps to array positions for O(log n) lookup
    pub index: BTreeMap<OffsetDateTime, usize>,
}

impl FinancialSeries {
    /// Create a new FinancialSeries from timestamps and values
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Vector of ordered timestamps
    /// * `values` - Array of values corresponding to timestamps
    ///
    /// # Panics
    ///
    /// Panics if timestamps and values have different lengths
    pub fn new(timestamps: Vec<OffsetDateTime>, values: Array1<f64>) -> Self {
        assert_eq!(
            timestamps.len(),
            values.len(),
            "Timestamps and values must have the same length"
        );

        // Build index for efficient timestamp lookup
        let index: BTreeMap<OffsetDateTime, usize> = timestamps
            .iter()
            .enumerate()
            .map(|(i, &timestamp)| (timestamp, i))
            .collect();

        Self {
            timestamps,
            values,
            index,
        }
    }

    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Calculate percentage changes over a specified number of periods
    ///
    /// This method efficiently computes percentage changes using the formula:
    /// `(current_value - previous_value) / previous_value`
    ///
    /// # Arguments
    ///
    /// * `periods` - Number of periods to look back for the calculation
    ///
    /// # Returns
    ///
    /// Array of percentage changes with NaN for the first `periods` values
    /// and where division by zero would occur
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rust_indicators::financial::FinancialSeries;
    /// # use time::OffsetDateTime;
    /// # use ndarray::Array1;
    /// let timestamps = vec![
    ///     OffsetDateTime::now_utc(),
    ///     OffsetDateTime::now_utc() + time::Duration::days(1),
    ///     OffsetDateTime::now_utc() + time::Duration::days(2),
    /// ];
    /// let values = Array1::from(vec![100.0, 102.0, 104.04]);
    /// let series = FinancialSeries::new(timestamps, values);
    ///
    /// let pct_changes = series.pct_change(1);
    /// // First value is NaN, second is 0.02 (2%), third is ~0.02 (2%)
    /// ```
    pub fn pct_change(&self, periods: usize) -> Array1<f64> {
        let mut result = Array1::zeros(self.values.len());

        // Fill first `periods` values with NaN
        for i in 0..periods.min(self.values.len()) {
            result[i] = f64::NAN;
        }

        // Calculate percentage changes for remaining values
        for i in periods..self.values.len() {
            let current = self.values[i];
            let previous = self.values[i - periods];

            if previous == 0.0 {
                result[i] = f64::NAN; // Avoid division by zero
            } else {
                result[i] = (current - previous) / previous;
            }
        }

        result
    }

    /// Create memory-efficient rolling windows over the values
    ///
    /// Returns an iterator that yields array views of the specified window size.
    /// This is a zero-copy operation that provides efficient access to rolling windows.
    ///
    /// # Arguments
    ///
    /// * `window` - Size of the rolling window
    ///
    /// # Returns
    ///
    /// Iterator yielding ArrayView1<f64> for each window
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rust_indicators::financial::FinancialSeries;
    /// # use time::OffsetDateTime;
    /// # use ndarray::Array1;
    /// let timestamps = vec![
    ///     OffsetDateTime::now_utc(),
    ///     OffsetDateTime::now_utc() + time::Duration::days(1),
    ///     OffsetDateTime::now_utc() + time::Duration::days(2),
    ///     OffsetDateTime::now_utc() + time::Duration::days(3),
    /// ];
    /// let values = Array1::from(vec![100.0, 102.0, 104.0, 103.0]);
    /// let series = FinancialSeries::new(timestamps, values);
    ///
    /// // Get 3-period rolling windows
    /// for window in series.rolling_window(3) {
    ///     let mean = window.mean().unwrap();
    ///     println!("Window mean: {}", mean);
    /// }
    /// ```
    pub fn rolling_window(&self, window: usize) -> impl Iterator<Item = ArrayView1<'_, f64>> + '_ {
        (0..=self.values.len().saturating_sub(window))
            .map(move |i| self.values.slice(s![i..i + window]))
    }

    /// Align this series with another series, handling missing data
    ///
    /// This method aligns two time series by their timestamps, returning arrays
    /// of values that correspond to the intersection of timestamps. Missing values
    /// are handled by forward-filling or using NaN.
    ///
    /// # Arguments
    ///
    /// * `other` - The other FinancialSeries to align with
    ///
    /// # Returns
    ///
    /// Tuple of (self_aligned, other_aligned) arrays with matching timestamps
    ///
    /// # Example
    ///
    /// ```rust
    /// # use rust_indicators::financial::FinancialSeries;
    /// # use time::OffsetDateTime;
    /// # use ndarray::Array1;
    /// let base_time = OffsetDateTime::now_utc();
    /// let timestamps1 = vec![base_time, base_time + time::Duration::days(2)];
    /// let values1 = Array1::from(vec![100.0, 104.0]);
    /// let series1 = FinancialSeries::new(timestamps1, values1);
    ///
    /// let timestamps2 = vec![base_time, base_time + time::Duration::days(1), base_time + time::Duration::days(2)];
    /// let values2 = Array1::from(vec![200.0, 202.0, 204.0]);
    /// let series2 = FinancialSeries::new(timestamps2, values2);
    ///
    /// let (aligned1, aligned2) = series1.align_with(&series2);
    /// // Returns values for common timestamps
    /// ```
    pub fn align_with(&self, other: &Self) -> (Array1<f64>, Array1<f64>) {
        // Find common timestamps
        let common_timestamps: Vec<OffsetDateTime> = self
            .timestamps
            .iter()
            .filter(|&ts| other.index.contains_key(ts))
            .copied()
            .collect();

        let mut self_aligned = Array1::zeros(common_timestamps.len());
        let mut other_aligned = Array1::zeros(common_timestamps.len());

        for (i, &timestamp) in common_timestamps.iter().enumerate() {
            if let Some(&self_idx) = self.index.get(&timestamp) {
                self_aligned[i] = self.values[self_idx];
            }
            if let Some(&other_idx) = other.index.get(&timestamp) {
                other_aligned[i] = other.values[other_idx];
            }
        }

        (self_aligned, other_aligned)
    }

    /// Calculate rolling volatility using a specified window
    ///
    /// # Arguments
    ///
    /// * `window` - Size of the rolling window for volatility calculation
    ///
    /// # Returns
    ///
    /// Array of rolling volatility values (standard deviation of returns)
    pub fn rolling_volatility(&self, window: usize) -> Array1<f64> {
        let returns = self.pct_change(1);
        let mut volatility = Array1::zeros(self.values.len());

        // Fill first values with NaN
        for i in 0..window {
            volatility[i] = f64::NAN;
        }

        // Calculate rolling volatility
        for i in window..self.values.len() {
            let window_returns = returns.slice(s![i - window + 1..=i]);

            // Filter out NaN values
            let valid_returns: Vec<f64> = window_returns
                .iter()
                .filter(|&&x| !x.is_nan())
                .copied()
                .collect();

            if valid_returns.len() > 1 {
                let mean = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;
                let variance = valid_returns
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (valid_returns.len() - 1) as f64;
                volatility[i] = variance.sqrt();
            } else {
                volatility[i] = f64::NAN;
            }
        }

        volatility
    }

    /// Get value at a specific timestamp
    ///
    /// # Arguments
    ///
    /// * `timestamp` - The timestamp to look up
    ///
    /// # Returns
    ///
    /// Option containing the value if the timestamp exists
    pub fn get_value_at(&self, timestamp: OffsetDateTime) -> Option<f64> {
        self.index.get(&timestamp).map(|&idx| self.values[idx])
    }

    /// Get a slice of values between two timestamps (inclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - Start timestamp (inclusive)
    /// * `end` - End timestamp (inclusive)
    ///
    /// # Returns
    ///
    /// ArrayView1 of values in the specified range
    pub fn slice_by_time(
        &self,
        start: OffsetDateTime,
        end: OffsetDateTime,
    ) -> Option<ArrayView1<'_, f64>> {
        let start_idx = self.index.get(&start)?;
        let end_idx = self.index.get(&end)?;

        if start_idx <= end_idx {
            Some(self.values.slice(s![*start_idx..=*end_idx]))
        } else {
            None
        }
    }

    /// Calculate simple moving average with specified window
    ///
    /// # Arguments
    ///
    /// * `window` - Size of the moving average window
    ///
    /// # Returns
    ///
    /// Array of moving average values
    pub fn simple_moving_average(&self, window: usize) -> Array1<f64> {
        let mut sma = Array1::zeros(self.values.len());

        // Fill first values with NaN
        for i in 0..window.saturating_sub(1) {
            sma[i] = f64::NAN;
        }

        // Calculate SMA using rolling windows
        for (i, window_view) in self.rolling_window(window).enumerate() {
            sma[i + window - 1] = window_view.mean().unwrap_or(f64::NAN);
        }

        sma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    fn create_test_series() -> FinancialSeries {
        let base_time = OffsetDateTime::now_utc();
        let timestamps = vec![
            base_time,
            base_time + Duration::days(1),
            base_time + Duration::days(2),
            base_time + Duration::days(3),
            base_time + Duration::days(4),
        ];
        let values = Array1::from(vec![100.0, 102.0, 104.04, 102.0, 105.0]);
        FinancialSeries::new(timestamps, values)
    }

    #[test]
    fn test_new_series() {
        let series = create_test_series();
        assert_eq!(series.len(), 5);
        assert!(!series.is_empty());
        assert_eq!(series.index.len(), 5);
    }

    #[test]
    fn test_pct_change() {
        let series = create_test_series();
        let pct_changes = series.pct_change(1);

        assert!(pct_changes[0].is_nan()); // First value should be NaN
        assert!((pct_changes[1] - 0.02).abs() < 1e-10); // 2% change
        assert!((pct_changes[2] - 0.02).abs() < 1e-6); // ~2% change
    }

    #[test]
    fn test_rolling_window() {
        let series = create_test_series();
        let windows: Vec<_> = series.rolling_window(3).collect();

        assert_eq!(windows.len(), 3); // 5 - 3 + 1 = 3 windows
        assert_eq!(windows[0].len(), 3);
        assert_eq!(windows[0][0], 100.0);
        assert_eq!(windows[0][2], 104.04);
    }

    #[test]
    fn test_align_with() {
        let base_time = OffsetDateTime::now_utc();
        let timestamps1 = vec![base_time, base_time + Duration::days(2)];
        let values1 = Array1::from(vec![100.0, 104.0]);
        let series1 = FinancialSeries::new(timestamps1, values1);

        let timestamps2 = vec![
            base_time,
            base_time + Duration::days(1),
            base_time + Duration::days(2),
        ];
        let values2 = Array1::from(vec![200.0, 202.0, 204.0]);
        let series2 = FinancialSeries::new(timestamps2, values2);

        let (aligned1, aligned2) = series1.align_with(&series2);

        assert_eq!(aligned1.len(), 2); // Two common timestamps
        assert_eq!(aligned2.len(), 2);
        assert_eq!(aligned1[0], 100.0);
        assert_eq!(aligned2[0], 200.0);
        assert_eq!(aligned1[1], 104.0);
        assert_eq!(aligned2[1], 204.0);
    }

    #[test]
    fn test_rolling_volatility() {
        let series = create_test_series();
        let volatility = series.rolling_volatility(3);

        assert!(volatility[0].is_nan());
        assert!(volatility[1].is_nan());
        assert!(volatility[2].is_nan());
        assert!(!volatility[3].is_nan());
        assert!(volatility[3] > 0.0);
    }

    #[test]
    fn test_get_value_at() {
        let series = create_test_series();
        let timestamp = series.timestamps[2];

        assert_eq!(series.get_value_at(timestamp), Some(104.04));

        let non_existent = OffsetDateTime::now_utc() + Duration::days(100);
        assert_eq!(series.get_value_at(non_existent), None);
    }

    #[test]
    fn test_simple_moving_average() {
        let series = create_test_series();
        let sma = series.simple_moving_average(3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!(!sma[2].is_nan());

        // Check calculation: (100 + 102 + 104.04) / 3
        let expected = (100.0 + 102.0 + 104.04) / 3.0;
        assert!((sma[2] - expected).abs() < 1e-10);
    }
}
