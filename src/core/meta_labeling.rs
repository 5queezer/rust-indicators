use ndarray::Array1;

/// Implements the meta-labeling strategy for improving precision.
pub struct MetaLabeler {
    pub primary_threshold: f64,
    pub volatility_adjustment: bool,
}

impl MetaLabeler {
    pub fn new(primary_threshold: f64, volatility_adjustment: bool) -> Self {
        Self {
            primary_threshold,
            volatility_adjustment,
        }
    }

    pub fn generate_meta_labels(
        &self,
        primary_predictions: &Array1<f64>,
        actual_returns: &Array1<f64>,
        volatility: &Array1<f64>,
    ) -> Array1<i32> {
        primary_predictions.iter()
            .zip(actual_returns.iter())
            .zip(volatility.iter())
            .map(|((pred, ret), vol)| {
                let adjusted_threshold = if self.volatility_adjustment {
                    self.primary_threshold * vol
                } else {
                    self.primary_threshold
                };

                if pred.abs() > adjusted_threshold && pred.signum() == ret.signum() {
                    1
                } else {
                    0
                }
            })
            .collect::<Vec<_>>()
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_generate_meta_labels() {
        let labeler = MetaLabeler::new(0.5, false);
        let primary_predictions = array![0.6, 0.4, -0.7, 0.8];
        let actual_returns = array![1.0, 1.0, -1.0, -1.0];
        let volatility = array![1.0, 1.0, 1.0, 1.0];

        let labels = labeler.generate_meta_labels(&primary_predictions, &actual_returns, &volatility);
        assert_eq!(labels, array![1, 0, 1, 0]);
    }

    #[test]
    fn test_generate_meta_labels_with_volatility_adjustment() {
        let labeler = MetaLabeler::new(0.5, true);
        let primary_predictions = array![0.6, 0.4, -0.7, 0.8];
        let actual_returns = array![1.0, 1.0, -1.0, -1.0];
        let volatility = array![0.5, 2.0, 1.0, 0.5];

        // thresholds: 0.25, 1.0, 0.5, 0.25
        // pred.abs > threshold: 0.6>0.25 (T), 0.4>1.0 (F), 0.7>0.5 (T), 0.8>0.25 (T)
        // signum match: T, T, T, F
        // result: 1, 0, 1, 0

        let labels = labeler.generate_meta_labels(&primary_predictions, &actual_returns, &volatility);
        assert_eq!(labels, array![1, 0, 1, 0]);
    }
}
