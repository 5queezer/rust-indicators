use std::collections::{BTreeMap, BTreeSet};
use ndarray::{Array1, ArrayView1};
use time::{Duration, OffsetDateTime};

/// A time series data structure for financial data, optimized for performance.
///
/// This struct holds time series data, with timestamps and corresponding values.
/// It includes a B-Tree map for fast lookups of timestamps to indices.
#[derive(Debug, Clone)]
pub struct FinancialSeries {
    pub timestamps: Vec<OffsetDateTime>,
    pub values: Array1<f64>,
    pub index: BTreeMap<OffsetDateTime, usize>,
}

impl FinancialSeries {
    /// Creates a new `FinancialSeries` from a vector of timestamps and values.
    ///
    /// The index is automatically built from the timestamps.
    pub fn new(timestamps: Vec<OffsetDateTime>, values: Array1<f64>) -> Self {
        let index = timestamps
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i))
            .collect();
        Self {
            timestamps,
            values,
            index,
        }
    }

    /// Calculates the percentage change between the current and a prior element.
    pub fn pct_change(&self, periods: usize) -> Array1<f64> {
        if periods == 0 {
            return Array1::from_elem(self.values.len(), 0.0);
        }
        if periods >= self.values.len() {
            return Array1::from_elem(self.values.len(), f64::NAN);
        }

        let mut result = Array1::from_elem(self.values.len(), f64::NAN);
        for i in periods..self.values.len() {
            let old_value = self.values[i - periods];
            if old_value != 0.0 {
                result[i] = (self.values[i] - old_value) / old_value;
            } else {
                result[i] = f64::NAN;
            }
        }
        result
    }

    /// Returns an iterator over a rolling window of a given size.
    ///
    /// This is memory-efficient as it returns views into the original data.
    pub fn rolling_window(&self, window: usize) -> impl Iterator<Item = ArrayView1<'_, f64>> {
        self.values.windows((window,)).into_iter()
    }

    /// Resamples the time series to a new frequency.
    ///
    /// This method groups the data by the given time `rule` (e.g., daily, weekly)
    /// and aggregates the values in each group. Currently, it uses the 'last'
    /// value in each group for aggregation.
    ///
    /// # Arguments
    ///
    /// * `rule` - The time duration to resample by.
    ///
    /// # Returns
    ///
    /// A new `FinancialSeries` with the resampled data.
    pub fn resample(&self, rule: Duration) -> Self {
        if self.timestamps.is_empty() {
            return Self::new(vec![], Array1::from(vec![]));
        }

        let mut resampled_data: BTreeMap<OffsetDateTime, Vec<f64>> = BTreeMap::new();

        let rule_seconds = rule.as_seconds_f64();
        if rule_seconds <= 0.0 {
            // Return a clone of the original series if the rule is not positive.
            return self.clone();
        }

        for (i, &timestamp) in self.timestamps.iter().enumerate() {
            let unix_ts = timestamp.unix_timestamp();
            // Calculate the start of the time bin.
            let bin_start_unix_ts = (unix_ts as f64 / rule_seconds).floor() * rule_seconds;
            let bin_start_timestamp =
                OffsetDateTime::from_unix_timestamp(bin_start_unix_ts as i64).unwrap();

            resampled_data
                .entry(bin_start_timestamp)
                .or_default()
                .push(self.values[i]);
        }

        let mut resampled_timestamps = vec![];
        let mut resampled_values = vec![];

        for (timestamp, values) in resampled_data {
            resampled_timestamps.push(timestamp);
            // Use 'last' aggregation.
            if let Some(last_value) = values.last() {
                resampled_values.push(*last_value);
            }
        }

        Self::new(resampled_timestamps, Array1::from(resampled_values))
    }

    /// Applies a function to each value in the series, returning a new series.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes an `f64` and returns an `f64`.
    ///
    /// # Returns
    ///
    /// A new `FinancialSeries` with the transformed values.
    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let new_values = self.values.mapv(f);
        Self::new(self.timestamps.clone(), new_values)
    }

    /// Aligns this series with another, handling missing data.
    pub fn align_with(&self, other: &Self) -> (Array1<f64>, Array1<f64>) {
        let all_timestamps: BTreeSet<_> = self
            .timestamps
            .iter()
            .chain(other.timestamps.iter())
            .collect();

        let mut aligned_self_values = Vec::with_capacity(all_timestamps.len());
        let mut aligned_other_values = Vec::with_capacity(all_timestamps.len());

        for &timestamp in all_timestamps.iter() {
            let self_value = self
                .index
                .get(timestamp)
                .map_or(f64::NAN, |&i| self.values[i]);
            let other_value = other
                .index
                .get(timestamp)
                .map_or(f64::NAN, |&i| other.values[i]);

            aligned_self_values.push(self_value);
            aligned_other_values.push(other_value);
        }

        (
            Array1::from(aligned_self_values),
            Array1::from(aligned_other_values),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use time::macros::datetime;

    fn create_test_series() -> FinancialSeries {
        let timestamps = vec![
            datetime!(2023-01-01 0:00 UTC),
            datetime!(2023-01-02 0:00 UTC),
            datetime!(2023-01-03 0:00 UTC),
            datetime!(2023-01-04 0:00 UTC),
            datetime!(2023-01-05 0:00 UTC),
        ];
        let values = array![10.0, 12.0, 15.0, 14.0, 15.0];
        FinancialSeries::new(timestamps, values)
    }

    #[test]
    fn test_new_financial_series() {
        let series = create_test_series();
        assert_eq!(series.values.len(), 5);
        assert_eq!(series.timestamps.len(), 5);
        assert_eq!(series.index.len(), 5);
        assert_eq!(*series.index.get(&datetime!(2023-01-01 0:00 UTC)).unwrap(), 0);
        assert_eq!(*series.index.get(&datetime!(2023-01-05 0:00 UTC)).unwrap(), 4);
    }

    #[test]
    fn test_pct_change() {
        let series = create_test_series();
        let changes = series.pct_change(1);

        assert!(changes[0].is_nan());
        assert!((changes[1] - 0.2).abs() < 1e-9);
        assert!((changes[2] - 0.25).abs() < 1e-9);
        assert!((changes[3] - (-0.06666666666666667)).abs() < 1e-9);
        assert!((changes[4] - 0.07142857142857142).abs() < 1e-9);
    }

    #[test]
    fn test_pct_change_with_zero() {
        let timestamps = vec![
            datetime!(2023-01-01 0:00 UTC),
            datetime!(2023-01-02 0:00 UTC),
        ];
        let values = array![0.0, 10.0];
        let series = FinancialSeries::new(timestamps, values);
        let changes = series.pct_change(1);
        assert!(changes[0].is_nan());
        assert!(changes[1].is_nan());
    }

    #[test]
    fn test_rolling_window() {
        let series = create_test_series();
        let mut windows = series.rolling_window(3);

        assert_eq!(windows.next().unwrap(), array![10.0, 12.0, 15.0]);
        assert_eq!(windows.next().unwrap(), array![12.0, 15.0, 14.0]);
        assert_eq!(windows.next().unwrap(), array![15.0, 14.0, 15.0]);
        assert!(windows.next().is_none());
    }

    #[test]
    fn test_resample() {
        let timestamps = vec![
            datetime!(2023-01-01 12:00 UTC),
            datetime!(2023-01-02 12:00 UTC),
            datetime!(2023-01-03 12:00 UTC),
            datetime!(2023-01-04 12:00 UTC),
            datetime!(2023-01-05 12:00 UTC),
        ];
        let values = array![10.0, 12.0, 15.0, 14.0, 16.0];
        let series = FinancialSeries::new(timestamps, values);

        let resampled_series = series.resample(Duration::days(2));

        let expected_timestamps = vec![
            datetime!(2023-01-01 0:00 UTC),
            datetime!(2023-01-03 0:00 UTC),
            datetime!(2023-01-05 0:00 UTC),
        ];
        let expected_values = array![12.0, 14.0, 16.0]; // last value in each bin

        assert_eq!(resampled_series.timestamps, expected_timestamps);
        assert_eq!(resampled_series.values, expected_values);
    }

    #[test]
    fn test_apply() {
        let series = create_test_series();
        let new_series = series.apply(|x| x * 2.0);

        let expected_values = array![20.0, 24.0, 30.0, 28.0, 30.0];
        assert_eq!(new_series.values, expected_values);
        assert_eq!(new_series.timestamps, series.timestamps);
    }

    #[test]
    fn test_align_with() {
        let timestamps1 = vec![
            datetime!(2023-01-01 0:00 UTC),
            datetime!(2023-01-02 0:00 UTC),
            datetime!(2023-01-04 0:00 UTC),
        ];
        let values1 = array![1.0, 2.0, 4.0];
        let series1 = FinancialSeries::new(timestamps1, values1);

        let timestamps2 = vec![
            datetime!(2023-01-02 0:00 UTC),
            datetime!(2023-01-03 0:00 UTC),
            datetime!(2023-01-04 0:00 UTC),
        ];
        let values2 = array![20.0, 30.0, 40.0];
        let series2 = FinancialSeries::new(timestamps2, values2);

        let (aligned1, aligned2) = series1.align_with(&series2);

        // Expected union of timestamps is 2023-01-01, 2023-01-02, 2023-01-03, 2023-01-04
        let expected_values1 = array![1.0, 2.0, f64::NAN, 4.0];
        let expected_values2 = array![f64::NAN, 20.0, 30.0, 40.0];

        // Check for NaN equality separately
        assert_eq!(aligned1.len(), 4);
        assert_eq!(aligned1[0], expected_values1[0]);
        assert_eq!(aligned1[1], expected_values1[1]);
        assert!(aligned1[2].is_nan());
        assert_eq!(aligned1[3], expected_values1[3]);

        assert_eq!(aligned2.len(), 4);
        assert!(aligned2[0].is_nan());
        assert_eq!(aligned2[1], expected_values2[1]);
        assert_eq!(aligned2[2], expected_values2[2]);
        assert_eq!(aligned2[3], expected_values2[3]);
    }
}
