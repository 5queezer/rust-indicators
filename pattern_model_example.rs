use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use candle_core::{Device, Tensor, DType};
use std::collections::HashMap;

#[pyclass]
pub struct PatternRecognitionClassifier {
    device: Device,
    pattern_weights: Option<HashMap<String, f32>>,
    ensemble_models: Vec<Tensor>,
    
    cv_splits: Vec<(Vec<usize>, Vec<usize>)>,
    sample_weights: Vec<f32>,
    pattern_importance: HashMap<String, f32>,
    
    trained: bool,
    pattern_names: Vec<String>,
    confidence_threshold: f32,
}

#[pymethods]
impl PatternRecognitionClassifier {
    #[new]
    fn new(pattern_names: Vec<String>) -> PyResult<Self> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

        Ok(PatternRecognitionClassifier {
            device,
            pattern_weights: None,
            ensemble_models: Vec::new(),
            cv_splits: Vec::new(),
            sample_weights: Vec::new(),
            pattern_importance: HashMap::new(),
            trained: false,
            pattern_names,
            confidence_threshold: 0.6,
        })
    }

    fn create_pattern_labels(
        &self,
        py: Python,
        open_prices: PyReadonlyArray1<f32>,
        high_prices: PyReadonlyArray1<f32>,
        low_prices: PyReadonlyArray1<f32>,
        close_prices: PyReadonlyArray1<f32>,
        future_periods: usize,
        profit_threshold: f32,
        stop_threshold: f32,
    ) -> PyResult<Py<PyArray1<i32>>> {
        let opens = open_prices.as_slice()?;
        let highs = high_prices.as_slice()?;
        let lows = low_prices.as_slice()?;
        let closes = close_prices.as_slice()?;
        let n = opens.len();
        let mut labels = vec![1i32; n];
        
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
            
            if signal == 1 {
                let final_price = closes[(i + future_periods).min(n - 1)];
                let final_return = (final_price / entry_price) - 1.0;
                signal = if final_return > 0.005 { 2 } else if final_return < -0.005 { 0 } else { 1 };
            }
            
            labels[i] = signal;
        }
        
        Ok(PyArray1::from_vec(py, labels).to_owned().into())
    }

    fn calculate_pattern_sample_weights(
        &mut self,
        pattern_signals: PyReadonlyArray2<f32>,
        volatility: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let signals = pattern_signals.as_array();
        let vols = volatility.as_slice()?;
        let n = signals.nrows();
        
        self.sample_weights = vec![1.0; n];
        
        for i in 0..n {
            let pattern_row = signals.row(i);
            let pattern_count = pattern_row.iter().filter(|&&x| x > 0.5).count() as f32;
            let vol_weight = (vols[i] / 0.02).clamp(0.5, 2.0); // Normalize around 2% volatility
            
            // Weight by pattern rarity and market volatility
            let rarity_weight = if pattern_count > 0.0 { 1.0 / pattern_count.sqrt() } else { 1.0 };
            self.sample_weights[i] = rarity_weight * vol_weight;
        }
        
        Ok(())
    }

    fn create_purged_pattern_cv(&mut self, n_samples: usize, pattern_duration: usize) -> PyResult<()> {
        self.cv_splits.clear();
        let n_splits = 3;
        let fold_size = n_samples / n_splits;
        let embargo = (pattern_duration * 2).max(10); // Pattern-aware embargo
        
        for fold in 0..n_splits {
            let test_start = fold * fold_size;
            let test_end = if fold == n_splits - 1 { n_samples } else { (fold + 1) * fold_size };
            
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let mut train_indices = Vec::new();
            
            if test_start > embargo {
                train_indices.extend(0..(test_start - embargo));
            }
            
            if test_end + embargo < n_samples {
                train_indices.extend((test_end + embargo)..n_samples);
            }
            
            self.cv_splits.push((train_indices, test_indices));
        }
        
        Ok(())
    }

    fn train_pattern_ensemble(
        &mut self,
        pattern_features: PyReadonlyArray2<f32>,
        price_features: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<i32>,
        pattern_names: Vec<String>,
    ) -> PyResult<HashMap<String, f32>> {
        let pattern_array = pattern_features.as_array();
        let price_array = price_features.as_array();
        let y_array = y.as_array();
        let (n_samples, n_patterns) = pattern_array.dim();
        let (_, n_price_features) = price_array.dim();
        
        if self.sample_weights.len() != n_samples {
            self.sample_weights = vec![1.0; n_samples];
        }
        
        self.create_purged_pattern_cv(n_samples, 5)?; // 5-bar pattern duration
        self.pattern_names = pattern_names;
        
        let mut cv_scores = Vec::new();
        let mut pattern_performances = HashMap::new();
        
        // Initialize pattern importance
        for pattern_name in &self.pattern_names {
            self.pattern_importance.insert(pattern_name.clone(), 0.0);
        }
        
        for (fold_idx, (train_idx, test_idx)) in self.cv_splits.iter().enumerate() {
            let fold_score = self.train_pattern_fold(
                &pattern_array, 
                &price_array, 
                &y_array, 
                train_idx, 
                test_idx
            )?;
            
            cv_scores.push(fold_score);
            
            // Calculate individual pattern performance
            for (i, pattern_name) in self.pattern_names.iter().enumerate() {
                if i < n_patterns {
                    let pattern_score = self.evaluate_single_pattern(
                        &pattern_array, &y_array, i, test_idx
                    )?;
                    
                    let current_score = pattern_performances.get(pattern_name).unwrap_or(&0.0);
                    pattern_performances.insert(pattern_name.clone(), current_score + pattern_score);
                }
            }
        }
        
        // Average pattern performances and update importance
        for (pattern_name, total_score) in pattern_performances {
            let avg_score = total_score / self.cv_splits.len() as f32;
            self.pattern_importance.insert(pattern_name, avg_score);
        }
        
        // Create ensemble weights
        self.pattern_weights = Some(self.calculate_pattern_weights()?);
        self.trained = true;
        
        let mean_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let std_score = {
            let variance = cv_scores.iter()
                .map(|&x| (x - mean_score).powi(2))
                .sum::<f32>() / cv_scores.len() as f32;
            variance.sqrt()
        };
        
        let mut results = HashMap::new();
        results.insert("cv_mean".to_string(), mean_score);
        results.insert("cv_std".to_string(), std_score);
        results.insert("n_patterns".to_string(), self.pattern_names.len() as f32);
        results.insert("best_pattern_score".to_string(), 
            self.pattern_importance.values().fold(0.0f32, |a, &b| a.max(b)));
        
        Ok(results)
    }

    fn predict_pattern_ensemble(
        &self,
        py: Python,
        pattern_features: PyReadonlyArray1<f32>,
        price_features: PyReadonlyArray1<f32>,
    ) -> PyResult<(i32, f32, Py<PyArray1<f32>>)> {
        if !self.trained {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not trained"));
        }
        
        let patterns = pattern_features.as_slice()?;
        let prices = price_features.as_slice()?;
        let weights = self.pattern_weights.as_ref().unwrap();
        
        let mut weighted_signals = Vec::new();
        let mut total_confidence = 0.0;
        let mut active_patterns = 0;
        
        // Calculate weighted ensemble prediction
        for (i, &pattern_signal) in patterns.iter().enumerate() {
            if i < self.pattern_names.len() && pattern_signal > 0.1 {
                let pattern_name = &self.pattern_names[i];
                if let Some(&weight) = weights.get(pattern_name) {
                    weighted_signals.push(pattern_signal * weight);
                    total_confidence += weight * pattern_signal;
                    active_patterns += 1;
                }
            }
        }
        
        // Combine with price momentum
        let price_momentum = if prices.len() >= 2 {
            (prices[prices.len()-1] / prices[prices.len()-2]) - 1.0
        } else { 0.0 };
        
        let pattern_signal = if !weighted_signals.is_empty() {
            weighted_signals.iter().sum::<f32>() / weighted_signals.len() as f32
        } else { 0.0 };
        
        let combined_signal = 0.7 * pattern_signal + 0.3 * price_momentum.tanh();
        
        let prediction = if combined_signal > 0.15 {
            2 // Strong buy
        } else if combined_signal < -0.15 {
            0 // Strong sell
        } else {
            1 // Hold
        };
        
        let confidence = if active_patterns > 0 {
            (total_confidence / active_patterns as f32).min(1.0)
        } else { 0.0 };
        
        // Return pattern contributions for interpretability
        let mut pattern_contributions = vec![0.0f32; self.pattern_names.len()];
        for (i, pattern_name) in self.pattern_names.iter().enumerate() {
            if i < patterns.len() && patterns[i] > 0.1 {
                if let Some(&weight) = weights.get(pattern_name) {
                    pattern_contributions[i] = patterns[i] * weight;
                }
            }
        }
        
        Ok((
            prediction, 
            confidence,
            PyArray1::from_vec(py, pattern_contributions).to_owned().into()
        ))
    }

    fn get_pattern_importance(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        let importance_vec: Vec<f32> = self.pattern_names.iter()
            .map(|name| self.pattern_importance.get(name).unwrap_or(&0.0).clone())
            .collect();
        
        Ok(PyArray1::from_vec(py, importance_vec).to_owned().into())
    }

    fn get_pattern_names(&self) -> Vec<String> {
        self.pattern_names.clone()
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

impl PatternRecognitionClassifier {
    fn train_pattern_fold(
        &self,
        pattern_features: &ndarray::ArrayView2<f32>,
        price_features: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        train_idx: &[usize],
        test_idx: &[usize],
    ) -> candle_core::Result<f32> {
        let mut correct = 0;
        let mut total = 0;
        
        for &idx in test_idx {
            if idx < pattern_features.nrows() && idx < y.len() {
                let pattern_row = pattern_features.row(idx).to_vec();
                let price_row = price_features.row(idx).to_vec();
                
                let (pred_class, confidence, _) = self.predict_sample(&pattern_row, &price_row)?;
                
                if confidence > self.confidence_threshold {
                    if pred_class == y[idx] {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }
        
        Ok(if total > 0 { correct as f32 / total as f32 } else { 0.0 })
    }

    fn predict_sample(&self, pattern_features: &[f32], price_features: &[f32]) -> candle_core::Result<(i32, f32, Vec<f32>)> {
        let pattern_signal = pattern_features.iter().sum::<f32>() / pattern_features.len() as f32;
        
        let price_momentum = if price_features.len() >= 2 {
            let recent_momentum = price_features[price_features.len()-1] - price_features[price_features.len()-2];
            recent_momentum.tanh()
        } else { 0.0 };
        
        let combined_signal = 0.6 * pattern_signal + 0.4 * price_momentum;
        let confidence = pattern_signal.abs().min(1.0);
        
        let prediction = if combined_signal > 0.2 {
            2
        } else if combined_signal < -0.2 {
            0
        } else {
            1
        };
        
        Ok((prediction, confidence, pattern_features.to_vec()))
    }

    fn evaluate_single_pattern(
        &self,
        pattern_features: &ndarray::ArrayView2<f32>,
        y: &ndarray::ArrayView1<i32>,
        pattern_idx: usize,
        test_idx: &[usize],
    ) -> candle_core::Result<f32> {
        let mut correct = 0;
        let mut pattern_signals = 0;
        
        for &idx in test_idx {
            if idx < pattern_features.nrows() && pattern_idx < pattern_features.ncols() {
                let pattern_signal = pattern_features[[idx, pattern_idx]];
                
                if pattern_signal > 0.5 { // Pattern detected
                    pattern_signals += 1;
                    let predicted_direction = if pattern_signal > 0.7 { 2 } else { 1 };
                    
                    if predicted_direction == y[idx] {
                        correct += 1;
                    }
                }
            }
        }
        
        Ok(if pattern_signals > 0 { 
            correct as f32 / pattern_signals as f32 
        } else { 0.0 })
    }

    fn calculate_pattern_weights(&self) -> candle_core::Result<HashMap<String, f32>> {
        let mut weights = HashMap::new();
        let total_importance: f32 = self.pattern_importance.values().sum();
        
        if total_importance > 0.0 {
            for (pattern_name, &importance) in &self.pattern_importance {
                weights.insert(pattern_name.clone(), importance / total_importance);
            }
        } else {
            let uniform_weight = 1.0 / self.pattern_names.len() as f32;
            for pattern_name in &self.pattern_names {
                weights.insert(pattern_name.clone(), uniform_weight);
            }
        }
        
        Ok(weights)
    }
}

#[pymodule]
fn pattern_recognition_ml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PatternRecognitionClassifier>()?;
    Ok(())
}