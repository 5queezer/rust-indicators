use ndarray::Array1;
use rayon::prelude::*;
use time::OffsetDateTime;
use crate::core::FinancialSeries;

/// Configuration for the Triple Barrier Method.
#[derive(Debug, Clone)]
pub struct TripleBarrierConfig {
    pub profit_mult: f64,
    pub stop_mult: f64,
    pub max_hold: usize,
    pub min_return: f64,
}

/// Represents an event for the Triple Barrier Method.
#[derive(Debug, Clone)]
pub struct Event {
    pub timestamp: OffsetDateTime,
    pub t1: Option<OffsetDateTime>,
    pub target: f64,
    pub side: f64,
}

/// A labeler that uses the Triple Barrier Method.
pub struct TripleBarrierLabeler {
    pub config: TripleBarrierConfig,
}

impl TripleBarrierLabeler {
    pub fn new(config: TripleBarrierConfig) -> Self {
        Self { config }
    }

    pub fn generate_labels_parallel(
        &self,
        prices: &FinancialSeries,
        events: &[Event],
    ) -> Array1<i32> {
        events.par_iter()
            .map(|event| self.label_single_event(prices, event))
            .collect::<Vec<_>>()
            .into()
    }

    fn label_single_event(&self, prices: &FinancialSeries, event: &Event) -> i32 {
        let start_idx = prices.index[&event.timestamp];
        let entry_price = prices.values[start_idx];

        let profit_target = entry_price * (1.0 + self.config.profit_mult * event.target);
        let stop_target = entry_price * (1.0 - self.config.stop_mult * event.target);

        let end_idx = match event.t1 {
            Some(t1) => prices.index.get(&t1).copied().unwrap_or(prices.values.len() - 1),
            None => (start_idx + self.config.max_hold).min(prices.values.len() - 1),
        };

        for i in (start_idx + 1)..=end_idx {
            let price = prices.values[i];
            let adjusted_price = price * event.side;
            let adjusted_profit = profit_target * event.side;
            let adjusted_stop = stop_target * event.side;

            if adjusted_price >= adjusted_profit {
                return 2;
            } else if adjusted_price <= adjusted_stop {
                return 0;
            }
        }

        let final_return = (prices.values[end_idx] / entry_price - 1.0) * event.side;
        if final_return > self.config.min_return {
            2
        } else if final_return < -self.config.min_return {
            0
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FinancialSeries;
    use ndarray::array;
    use time::macros::datetime;

    fn setup_test_data() -> (TripleBarrierLabeler, FinancialSeries, Event) {
        let config = TripleBarrierConfig {
            profit_mult: 0.03,
            stop_mult: 0.02,
            max_hold: 10,
            min_return: 0.001,
        };
        let labeler = TripleBarrierLabeler::new(config);

        let timestamps = vec![
            datetime!(2023-01-01 0:00 UTC),
            datetime!(2023-01-02 0:00 UTC),
            datetime!(2023-01-03 0:00 UTC),
            datetime!(2023-01-04 0:00 UTC),
            datetime!(2023-01-05 0:00 UTC),
        ];
        let values = array![100.0, 101.0, 102.0, 105.0, 103.0];
        let prices = FinancialSeries::new(timestamps, values);

        let event = Event {
            timestamp: prices.timestamps[0],
            t1: Some(prices.timestamps[4]),
            target: 1.0,
            side: 1.0, // Long position
        };

        (labeler, prices, event)
    }

    #[test]
    fn test_profit_target_hit() {
        let (labeler, prices, event) = setup_test_data();
        let label = labeler.label_single_event(&prices, &event);
        assert_eq!(label, 2); // Profit target hit at 105.0
    }

    #[test]
    fn test_stop_loss_hit() {
        let (labeler, mut prices, event) = setup_test_data();
        prices.values = array![100.0, 99.0, 98.0, 97.0, 99.0];
        let label = labeler.label_single_event(&prices, &event);
        assert_eq!(label, 0); // Stop loss hit at 97.0
    }

    #[test]
    fn test_time_barrier_hold() {
        let (labeler, mut prices, event) = setup_test_data();
        prices.values = array![100.0, 100.1, 100.2, 100.1, 100.0];
        let label = labeler.label_single_event(&prices, &event);
        assert_eq!(label, 1); // Time barrier hit, return is flat
    }

    #[test]
    fn test_time_barrier_buy() {
        let (labeler, mut prices, event) = setup_test_data();
        prices.values = array![100.0, 101.0, 100.5, 101.5, 102.0];
        let label = labeler.label_single_event(&prices, &event);
        assert_eq!(label, 2); // Time barrier hit, positive return
    }

    #[test]
    fn test_time_barrier_sell() {
        let (labeler, mut prices, event) = setup_test_data();
        prices.values = array![100.0, 99.0, 99.5, 98.5, 98.0];
        let label = labeler.label_single_event(&prices, &event);
        assert_eq!(label, 0); // Time barrier hit, negative return
    }
}
