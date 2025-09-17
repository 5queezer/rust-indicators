use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;

#[pyclass]
pub struct ScientificTradingClassifier {
    cv_splits: Vec<(Vec<usize>, Vec<usize>)>,
    embargo_pct: f32,
    feature_importance: Vec<f32>,
    sample_weights: Vec<f32>,
    model_weights: Vec<f32>,
    trained: bool,
    n_features: usize,
}

#[pymethods]
impl ScientificTradingClassifier {
    #[new]
    fn new(n_features: usize) -> PyResult<Self> {
        Ok(ScientificTradingClassifier {
            cv_splits: Vec::new(),
            embargo_pct: 0.01,
            feature_importance: vec![0.0; n_features],
            sample_weights: Vec::new(),
            model_weights: vec![0.0; n_features],
            trained: false,
            n_features,
        })
    }

    fn create_triple_barrier_labels(
        &self,
        py: Python,
        prices: PyReadonlyArray1<f32>,
        volatility: PyReadonlyArray1<f32>,
        profit_mult: f32,
        stop_mult: f32,
        max_hold: usize,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let prices = prices.as_slice()?;
        let vols = volatility.as_slice()?;
        
        if prices.len() != vols.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array length mismatch"));
        }
        
        let n = prices.len();
        let mut labels = vec![1i32; n];
        
        for i in 0..n.saturating_sub(max_hold) {
            let entry = prices[i];
            let vol = vols[i];
            
            if vol <= 0.0 || entry <= 0.0 {
                continue;
            }
            
            let profit_target = entry * (1.0 + profit_mult * vol);
            let stop_target = entry * (1.0 - stop_mult * vol);
            
            for j in (i + 1)..=(i + max_hold).min(n - 1) {
                let price = prices[j];
                
                if price >= profit_target {
                    labels[i] = 2;
                    break;
                } else if price <= stop_target {
                    labels[i] = 0;
                    break;
                } else if j == (i + max_hold).min(n - 1) {
                    let ret = (price / entry) - 1.0;
                    labels[i] = if ret > 0.002 { 2 } else if ret < -0.002 { 0 } else { 1 };
                }
            }
        }
        
        Ok(PyArray1::from_vec(py, labels).to_owned().into())
    }

    fn calculate_sample_weights(&mut self, returns: PyReadonlyArray1<f32>) -> PyResult<()> {
        let rets = returns.as_slice()?;
        let n = rets.len();
        self.sample_weights = vec![1.0; n];
        
        let window = 20.min(n);
        for i in window..n {
            let window_rets = &rets[(i.saturating_sub(window))..i];
            let abs_ret = rets[i].abs();
            let avg_abs_ret = window_rets.iter()
                .map(|r| r.abs())
                .sum::<f32>() / window_rets.len() as f32;
            
            if avg_abs_ret > 0.0 {
                self.sample_weights[i] = (abs_ret / avg_abs_ret).clamp(0.1, 3.0);
            }
        }
        
        Ok(())
    }

    fn create_purged_cv_splits(&mut self, n_samples: usize, n_splits: usize) -> PyResult<()> {
        if n_samples < n_splits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not enough samples for splits"));
        }
        
        self.cv_splits.clear();
        let fold_size = n_samples / n_splits;
        let embargo = ((n_samples as f32 * self.embargo_pct) as usize).max(1);
        
        for fold in 0..n_splits {
            let test_start = fold * fold_size;
            let test_end = if fold == n_splits - 1 { n_samples } else { (fold + 1) * fold_size };
            
            if test_start >= test_end {
                continue;
            }
            
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let mut train_indices = Vec::new();
            
            if test_start > embargo {
                train_indices.extend(0..(test_start - embargo));
            }
            
            if test_end + embargo < n_samples {
                train_indices.extend((test_end + embargo)..n_samples);
            }
            
            if !train_indices.is_empty() && !test_indices.is_empty() {
                self.cv_splits.push((train_indices, test_indices));
            }
        }
        
        if self.cv_splits.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("No valid CV splits created"));
        }
        
        Ok(())
    }

    fn train_scientific(
        &mut self,
        X: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        learning_rate: f32,
    ) -> PyResult<HashMap<String, f32>> {
        let X_array = X.as_array();
        let y_array = y.as_array();
        let (n_samples, n_features) = X_array.dim();
        
        if n_features != self.n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} features, got {}", self.n_features, n_features)
            ));
        }
        
        if n_samples != y_array.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("X and y length mismatch"));
        }
        
        if self.sample_weights.len() != n_samples {
            self.sample_weights = vec![1.0; n_samples];
        }
        
        self.create_purged_cv_splits(n_samples, 3)?;
        
        let mut cv_scores = Vec::new();
        let mut feature_scores = vec![0.0; n_features];
        
        for (train_idx, test_idx) in &self.cv_splits {
            let (fold_score, fold_feature_importance) = self.train_fold(
                &X_array, &y_array, train_idx, test_idx, learning_rate
            )?;
            
            cv_scores.push(fold_score);
            
            for i in 0..n_features {
                feature_scores[i] += fold_feature_importance[i];
            }
        }
        
        // Average feature importance across folds
        let n_folds = self.cv_splits.len() as f32;
        for i in 0..n_features {
            self.feature_importance[i] = feature_scores[i] / n_folds;
        }
        
        // Train final model on all data
        self.model_weights = self.train_final_model(&X_array, &y_array, learning_rate)?;
        self.trained = true;
        
        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let variance = cv_scores.iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f32>() / cv_scores.len() as f32;
        
        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), variance.sqrt());
        results.insert("n_folds".to_string(), cv_scores.len() as f32);
        
        Ok(results)
    }

    fn predict_with_confidence(
        &self, 
        py: Python, 
        features: PyReadonlyArray1<f32>
    ) -> PyResult<(i32, f32)> {
        if !self.trained {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not trained"));
        }
        
        let feats = features.as_slice()?;
        
        if feats.len() != self.n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} features, got {}", self.n_features, feats.len())
            ));
        }
        
        let (prediction, confidence) = self.predict_sample(feats)?;
        Ok((prediction, confidence))
    }

    fn get_feature_importance(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.feature_importance.clone()).to_owned().into())
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl ScientificTradingClassifier {
    fn train_fold(
        &self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        train_idx: &[usize],
        test_idx: &[usize],
        _lr: f32,
    ) -> Result<(f32, Vec<f32>), PyErr> {
        let mut correct = 0;
        let mut total = 0;
        let mut feature_correlations = vec![0.0; self.n_features];
        
        // Calculate feature importance on training set
        for feature_idx in 0..self.n_features {
            let mut feature_sum = 0.0;
            let mut label_sum = 0.0;
            let mut feature_label_sum = 0.0;
            let mut feature_sq_sum = 0.0;
            let mut label_sq_sum = 0.0;
            let train_size = train_idx.len() as f32;
            
            for &idx in train_idx {
                let feature_val = X[[idx, feature_idx]];
                let label_val = y[idx] as f32;
                
                feature_sum += feature_val;
                label_sum += label_val;
                feature_label_sum += feature_val * label_val;
                feature_sq_sum += feature_val * feature_val;
                label_sq_sum += label_val * label_val;
            }
            
            let feature_mean = feature_sum / train_size;
            let label_mean = label_sum / train_size;
            
            let numerator = feature_label_sum - train_size * feature_mean * label_mean;
            let denominator = ((feature_sq_sum - train_size * feature_mean * feature_mean) * 
                              (label_sq_sum - train_size * label_mean * label_mean)).sqrt();
            
            if denominator > 0.0 {
                feature_correlations[feature_idx] = (numerator / denominator).abs();
            }
        }
        
        // Test performance
        for &idx in test_idx {
            let features: Vec<f32> = X.row(idx).to_vec();
            let (pred_class, confidence) = self.predict_sample(&features)
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Prediction failed"))?;
            
            if confidence > 0.3 {
                if pred_class == y[idx] {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
        Ok((accuracy, feature_correlations))
    }

    fn predict_sample(&self, features: &[f32]) -> Result<(i32, f32), Box<dyn std::error::Error>> {
        if features.len() != self.model_weights.len() {
            return Err("Feature dimension mismatch".into());
        }
        
        let weighted_sum = features.iter()
            .zip(&self.model_weights)
            .map(|(f, w)| f * w)
            .sum::<f32>();
        
        let normalized = weighted_sum.tanh(); // Squash to [-1, 1]
        let confidence = normalized.abs().min(1.0);
        
        let prediction = if normalized > 0.15 {
            2 // Buy
        } else if normalized < -0.15 {
            0 // Sell  
        } else {
            1 // Hold
        };
        
        Ok((prediction, confidence))
    }

    fn train_final_model(
        &self,
        X: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        learning_rate: f32,
    ) -> Result<Vec<f32>, PyErr> {
        let (n_samples, n_features) = X.dim();
        let mut weights = vec![0.01; n_features]; // Small random initialization
        
        // Simple gradient descent for logistic regression
        let epochs = 100;
        
        for _ in 0..epochs {
            let mut gradient = vec![0.0; n_features];
            
            for i in 0..n_samples {
                let features: Vec<f32> = X.row(i).to_vec();
                let prediction = features.iter()
                    .zip(&weights)
                    .map(|(f, w)| f * w)
                    .sum::<f32>()
                    .tanh();
                
                let target = match y[i] {
                    0 => -1.0,
                    1 => 0.0,
                    2 => 1.0,
                    _ => 0.0,
                };
                
                let error = target - prediction;
                let weight_factor = self.sample_weights.get(i).unwrap_or(&1.0);
                
                for j in 0..n_features {
                    gradient[j] += error * features[j] * weight_factor;
                }
            }
            
            // Update weights
            for j in 0..n_features {
                weights[j] += learning_rate * gradient[j] / n_samples as f32;
            }
        }
        
        Ok(weights)
    }
}

#[pymodule]
fn trading_ml_scientific(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ScientificTradingClassifier>()?;
    Ok(())
}