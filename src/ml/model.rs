use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

/// Lightweight ML model for meta-labeling
#[pyclass]
pub struct RustMLModel {
    // Simple linear model weights for demonstration
    weights: Vec<f64>,
    bias: f64,
    trained: bool,
}

#[pymethods]
impl RustMLModel {
    #[new]
    fn new() -> Self {
        RustMLModel {
            weights: vec![0.1, -0.05, 0.15, 0.2, -0.1, 0.25, 0.3], // 7 features
            bias: 0.5,
            trained: false,
        }
    }

    /// Simple prediction using linear model + sigmoid activation
    fn predict(&self, py: Python, features: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let features = features.as_array();
        let n_features = self.weights.len();
        let n_samples = features.len() / n_features;

        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start_idx = i * n_features;
            let end_idx = start_idx + n_features;

            if end_idx <= features.len() {
                // Linear combination
                let mut linear_output = self.bias;
                for j in 0..n_features.min(end_idx - start_idx) {
                    let feature_val = features[start_idx + j];
                    if feature_val.is_finite() {
                        linear_output += self.weights[j] * feature_val;
                    }
                }

                // Sigmoid activation for probability output
                let probability = 1.0 / (1.0 + (-linear_output).exp());
                predictions.push(probability.clamp(0.01, 0.99)); // Avoid extreme values
            } else {
                predictions.push(0.5); // Neutral when insufficient features
            }
        }

        Ok(PyArray1::from_vec(py, predictions).to_owned().into())
    }

    /// Check if model is trained
    fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get model weights for inspection
    fn get_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        PyArray1::from_vec(py, self.weights.clone()).to_owned().into()
    }
}