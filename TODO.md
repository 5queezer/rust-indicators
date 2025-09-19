# **Rust-First Financial ML Framework TODO**

*Focus: Pure Rust implementation with minimal dependencies, leveraging insights from López de Prado's "Advances in Financial Machine Learning"*

## **PHASE 1: Core Financial Data Structures (Priority: Critical) COMPLETE**

### **1.1 Time Series Foundation**

```rust
// Dependencies
polars = { version = "0.35", features = ["lazy", "temporal", "strings"] }
ndarray = { version = "0.15", features = ["approx", "blas"] }
ndarray-stats = "0.5"
time = { version = "0.3", features = ["serde", "parsing"] }

// TODO: Core financial time series
#[derive(Debug, Clone)]
struct FinancialSeries {
    timestamps: Vec<OffsetDateTime>,
    values: Array1<f64>,
    index: BTreeMap<OffsetDateTime, usize>,
}

impl FinancialSeries {
    fn pct_change(&self, periods: usize) -> Array1<f64> {
        // Efficient percentage change calculation
    }
    
    fn rolling_window(&self, window: usize) -> impl Iterator<Item = ArrayView1<f64>> {
        // Memory-efficient rolling windows
    }
    
    fn align_with(&self, other: &Self) -> (Array1<f64>, Array1<f64>) {
        // Handle missing data alignment like pandas but faster
    }
}
```

**Sub-tasks:**

- [x] Implement zero-copy time series operations
- [x] Add business day calendar support using `time` crate
- [x] Memory-efficient rolling window iterators
- [x] Missing data handling strategies
- [x] Index alignment for multi-series operations

### **1.2 Bar Structures (Information-Driven Sampling)**

```rust
// TODO: Implement López de Prado's advanced bar types
#[derive(Debug, Clone)]
enum BarType {
    Time { frequency: Duration },
    Volume { threshold: u64 },
    Dollar { threshold: f64 },
    VolumeImbalance { threshold: f64 },
    TickImbalance { threshold: i32 },
    RunBars { run_length: usize },
}

#[derive(Debug, Clone)]
struct Bar {
    timestamp: OffsetDateTime,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
    vwap: f64,
    buy_volume: u64,
    sell_volume: u64,
}

struct BarBuilder {
    bar_type: BarType,
    current_bar: Option<Bar>,
    imbalance_tracker: ImbalanceTracker,
}

impl BarBuilder {
    fn process_tick(&mut self, tick: &Tick) -> Option<Bar> {
        // Smart sampling based on information arrival
    }
}
```

**Sub-tasks:**

- [x] Implement all 6 bar types from López de Prado
- [x] Add tick rule for trade classification
- [x] Volume imbalance tracking
- [x] Run detection algorithms
- [x] Memory-efficient streaming bar construction

## **PHASE 2: Fractional Differentiation (Priority: Critical) COMPLETE**

### **2.1 Memory-Preserving Stationarity**

```rust
// Dependencies
statrs = "0.16"  // For statistical tests

// TODO: Implement López de Prado's fractional differentiation
struct FractionalDifferentiator {
    d: f64,                    // Differentiation order
    threshold: f64,            // Weight cutoff (1e-5)
    weights: Vec<f64>,         // Pre-computed weights
    window_size: usize,        // Fixed window size
}

impl FractionalDifferentiator {
    fn new(d: f64, threshold: f64) -> Self {
        let weights = Self::compute_weights(d, threshold);
        Self {
            d,
            threshold,
            window_size: weights.len(),
            weights,
        }
    }
    
    fn compute_weights(d: f64, threshold: f64) -> Vec<f64> {
        // López de Prado's iterative weight computation
        let mut weights = vec![1.0];
        let mut k = 1;
        
        loop {
            let weight = -weights[k-1] * (d - k as f64 + 1.0) / k as f64;
            if weight.abs() < threshold { break; }
            weights.push(weight);
            k += 1;
        }
        weights
    }
    
    fn transform(&self, series: &Array1<f64>) -> Array1<f64> {
        // Fixed-width window fractional differentiation
        let n = series.len();
        let mut result = Array1::zeros(n);
        
        for i in self.window_size..n {
            let start = i - self.window_size + 1;
            let window = series.slice(s![start..=i]);
            result[i] = self.weights.iter()
                .zip(window.iter().rev())
                .map(|(w, x)| w * x)
                .sum();
        }
        result
    }
    
    fn find_optimal_d(series: &Array1<f64>, confidence_level: f64) -> f64 {
        // Binary search for minimum d that passes ADF test
        let mut low = 0.0;
        let mut high = 2.0;
        
        while high - low > 0.01 {
            let mid = (low + high) / 2.0;
            let differentiator = Self::new(mid, 1e-5);
            let diff_series = differentiator.transform(series);
            
            if Self::adf_test(&diff_series, confidence_level) {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    }
    
    fn adf_test(series: &Array1<f64>, confidence_level: f64) -> bool {
        // Augmented Dickey-Fuller test implementation
        // Return true if series is stationary at given confidence level
        todo!("Implement ADF test using statrs crate")
    }
}
```

**Sub-tasks:**

- [x] Implement López de Prado's fixed-width window method (Snippet 5.3)
- [x] Add ADF test for stationarity checking
- [x] Binary search for optimal d value
- [x] Memory correlation preservation measurement
- [x] Batch processing for multiple series

### **2.2 Stationarity Testing Suite**

```rust
// TODO: Comprehensive stationarity tests
struct StationarityTests;

impl StationarityTests {
    fn adf_test(series: &Array1<f64>) -> (f64, f64, bool) {
        // Returns (statistic, p_value, is_stationary)
    }
    
    fn kpss_test(series: &Array1<f64>) -> (f64, f64, bool) {
        // Kwiatkowski-Phillips-Schmidt-Shin test
    }
    
    fn phillips_perron_test(series: &Array1<f64>) -> (f64, f64, bool) {
        // Phillips-Perron unit root test
    }
}
```

## **PHASE 3: Scientific Labeling Methods (Priority: Critical) COMPLETE**

### **3.1 Triple Barrier Implementation**

```rust
use rayon::prelude::*;

// TODO: Complete triple barrier method
#[derive(Debug, Clone)]
struct TripleBarrierConfig {
    profit_mult: f64,
    stop_mult: f64,
    max_hold: usize,
    min_return: f64,  // For time-based exit classification
}

#[derive(Debug, Clone)]
struct Event {
    timestamp: OffsetDateTime,
    t1: Option<OffsetDateTime>,  // Time barrier (optional)
    target: f64,                 // Unit width of horizontal barriers
    side: f64,                   // Position side (1.0 for long, -1.0 for short)
}

struct TripleBarrierLabeler {
    config: TripleBarrierConfig,
}

impl TripleBarrierLabeler {
    fn generate_labels_parallel(
        &self,
        prices: &FinancialSeries,
        events: &[Event],
    ) -> Array1<i32> {
        // Parallel processing using rayon
        events.par_iter()
            .map(|event| self.label_single_event(prices, event))
            .collect::<Vec<_>>()
            .into()
    }
    
    fn label_single_event(&self, prices: &FinancialSeries, event: &Event) -> i32 {
        let start_idx = prices.index[&event.timestamp];
        let entry_price = prices.values[start_idx];
        
        // Calculate barriers
        let profit_target = entry_price * (1.0 + self.config.profit_mult * event.target);
        let stop_target = entry_price * (1.0 - self.config.stop_mult * event.target);
        
        // Determine end index
        let end_idx = match event.t1 {
            Some(t1) => prices.index.get(&t1).copied().unwrap_or(prices.values.len() - 1),
            None => (start_idx + self.config.max_hold).min(prices.values.len() - 1),
        };
        
        // Check path for barrier touches
        for i in (start_idx + 1)..=end_idx {
            let price = prices.values[i];
            let adjusted_price = price * event.side;
            let adjusted_profit = profit_target * event.side;
            let adjusted_stop = stop_target * event.side;
            
            if adjusted_price >= adjusted_profit {
                return 2; // Buy signal (profit target hit)
            } else if adjusted_price <= adjusted_stop {
                return 0; // Sell signal (stop loss hit)
            }
        }
        
        // Time barrier hit - classify by final return
        let final_return = (prices.values[end_idx] / entry_price - 1.0) * event.side;
        if final_return > self.config.min_return {
            2
        } else if final_return < -self.config.min_return {
            0
        } else {
            1 // Hold
        }
    }
}
```

**Sub-tasks:**

- [x] Support all 8 barrier configurations from López de Prado
- [x] Add side-aware labeling (long/short positions)
- [x] Implement parallel processing with rayon
- [x] Add proper path-dependent logic
- [x] Support for disabled barriers (None values)

### **3.2 Meta-Labeling Strategy**

```rust
// TODO: Implement meta-labeling for precision improvement
struct MetaLabeler {
    primary_threshold: f64,
    volatility_adjustment: bool,
}

impl MetaLabeler {
    fn generate_meta_labels(
        &self,
        primary_predictions: &Array1<f64>,
        actual_returns: &Array1<f64>,
        volatility: &Array1<f64>,
    ) -> Array1<i32> {
        // Binary classification: bet (1) or no bet (0)
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
                    1 // Correct prediction above threshold
                } else {
                    0 // No bet
                }
            })
            .collect::<Vec<_>>()
            .into()
    }
}
```

## **PHASE 4: Advanced Cross-Validation (Priority: Critical) COMPLETE**

### **4.1 Purged Cross-Validation**

```rust
// TODO: Implement López de Prado's purged cross-validation
struct PurgedKFold {
    n_splits: usize,
    embargo_pct: f64,
    label_endtimes: Vec<OffsetDateTime>,  // t1 from events
}

impl PurgedKFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / self.n_splits;
        let embargo_size = ((n_samples as f64) * self.embargo_pct) as usize;
        
        (0..self.n_splits)
            .map(|fold| {
                let test_start = fold * fold_size;
                let test_end = if fold == self.n_splits - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };
                
                let test_indices: Vec<usize> = (test_start..test_end).collect();
                let train_indices = self.create_purged_train_set(
                    n_samples,
                    &test_indices,
                    embargo_size,
                );
                
                (train_indices, test_indices)
            })
            .collect()
    }
    
    fn create_purged_train_set(
        &self,
        n_samples: usize,
        test_indices: &[usize],
        embargo_size: usize,
    ) -> Vec<usize> {
        let test_start = test_indices[0];
        let test_end = test_indices[test_indices.len() - 1];
        
        // Find maximum label end time in test set
        let max_test_endtime = test_indices.iter()
            .map(|&i| self.label_endtimes[i])
            .max()
            .unwrap();
        
        // Find overlapping training samples to purge
        let max_endtime_idx = self.label_endtimes.iter()
            .position(|&t| t >= max_test_endtime)
            .unwrap_or(n_samples);
        
        let mut train_indices = Vec::new();
        
        // Add training data before test set (with embargo)
        if test_start > embargo_size {
            train_indices.extend(0..(test_start - embargo_size));
        }
        
        // Add training data after test set (with embargo and purging)
        if max_endtime_idx + embargo_size < n_samples {
            train_indices.extend((max_endtime_idx + embargo_size)..n_samples);
        }
        
        train_indices
    }
}
```

**Sub-tasks:**

- [x] Implement proper overlap detection based on label end times
- [x] Add embargo periods to prevent information leakage
- [x] Support for different embargo strategies
- [x] Validation of cross-validation quality
- [x] Pattern-aware cross-validation for specific applications

### **4.2 Combinatorial Purged Cross-Validation**

```rust
// TODO: Advanced cross-validation for overfitting detection
struct CombinatorialPurgedCV {
    base_cv: PurgedKFold,
    n_combinations: usize,
}

impl CombinatorialPurgedCV {
    fn generate_combinations(&self, n_samples: usize) -> Vec<Vec<(Vec<usize>, Vec<usize>)>> {
        // Generate multiple CV configurations for robustness testing
    }
    
    fn backtest_overfitting_probability(&self, performance_distribution: &[f64]) -> f64 {
        // Calculate probability of backtest overfitting using Sharpe ratio distribution
    }
}
```

## **PHASE 5: Sample Weighting (Priority: High)**

### **5.1 Overlap-Based Weighting**

```rust
// TODO: Implement López de Prado's sample weighting
struct SampleWeightCalculator {
    label_endtimes: Vec<OffsetDateTime>,
    overlap_matrix: Array2<bool>,
}

impl SampleWeightCalculator {
    fn calculate_average_uniqueness(&self) -> Array1<f64> {
        let n = self.label_endtimes.len();
        let mut uniqueness = Array1::zeros(n);
        
        for i in 0..n {
            let overlaps: usize = (0..n)
                .map(|j| if self.overlap_matrix[[i, j]] { 1 } else { 0 })
                .sum();
            
            uniqueness[i] = 1.0 / overlaps.max(1) as f64;
        }
        
        uniqueness
    }
    
    fn sequential_bootstrap(&self, sample_length: usize) -> Vec<usize> {
        // López de Prado's sequential bootstrap for non-IID samples
        let uniqueness = self.calculate_average_uniqueness();
        let mut selected_indices = Vec::new();
        let mut prob_weights = uniqueness.clone();
        
        while selected_indices.len() < sample_length {
            let idx = self.weighted_random_choice(&prob_weights);
            selected_indices.push(idx);
            
            // Reduce probability of overlapping samples
            for j in 0..prob_weights.len() {
                if self.overlap_matrix[[idx, j]] {
                    prob_weights[j] *= 0.5; // Decay factor
                }
            }
        }
        
        selected_indices
    }
    
    fn weighted_random_choice(&self, weights: &Array1<f64>) -> usize {
        // Weighted random sampling
        use rand::Rng;
        let total: f64 = weights.sum();
        let mut rng = rand::thread_rng();
        let target = rng.gen::<f64>() * total;
        
        let mut cumsum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum >= target {
                return i;
            }
        }
        weights.len() - 1
    }
}
```

**Sub-tasks:**

- [ ] Implement concurrent label detection
- [ ] Calculate average uniqueness weights
- [ ] Sequential bootstrap sampling
- [ ] Time-decay weighting factors
- [ ] Volatility-based sample weighting

## **PHASE 6: Microstructural Features (Priority: Medium)**

### **6.1 External VPIN Library Integration**

```rust
// TODO: Find and integrate existing VPIN implementation
// Potential libraries to investigate:
// - quantlib-rs
// - ta-rs extensions
// - financial-ml crate (if exists)

// For now, simple implementation as fallback
struct VPINCalculator {
    window_size: usize,
}

impl VPINCalculator {
    fn calculate(&self, buy_volumes: &[f64], sell_volumes: &[f64]) -> f64 {
        let total_imbalance: f64 = buy_volumes.iter()
            .zip(sell_volumes.iter())
            .map(|(buy, sell)| (buy - sell).abs())
            .sum();
        
        let total_volume: f64 = buy_volumes.iter()
            .zip(sell_volumes.iter())
            .map(|(buy, sell)| buy + sell)
            .sum();
        
        if total_volume > 0.0 {
            total_imbalance / total_volume
        } else {
            0.0
        }
    }
}
```

### **6.2 Order Flow Features**

```rust
// Dependencies for microstructural analysis
derive_more = "0.99"

#[derive(Debug, Clone, Add, Sub, Mul, Div)]
struct OrderFlow {
    volume_imbalance: f64,
    tick_imbalance: i32,
    dollar_imbalance: f64,
}

struct MicrostructuralFeatures;

impl MicrostructuralFeatures {
    fn kyle_lambda(signed_volume: &Array1<f64>, price_changes: &Array1<f64>) -> f64 {
        // Price impact coefficient
        let covariance = Self::covariance(signed_volume, price_changes);
        let variance = Self::variance(signed_volume);
        
        if variance > 0.0 {
            covariance / variance
        } else {
            0.0
        }
    }
    
    fn amihud_lambda(returns: &Array1<f64>, dollar_volume: &Array1<f64>) -> Array1<f64> {
        // Illiquidity measure
        returns.iter()
            .zip(dollar_volume.iter())
            .map(|(ret, vol)| {
                if *vol > 0.0 {
                    ret.abs() / vol
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>()
            .into()
    }
    
    fn roll_spread_estimator(price_changes: &Array1<f64>) -> f64 {
        // Estimate bid-ask spread from price changes
        let autocovariance = Self::autocovariance(price_changes, 1);
        if autocovariance < 0.0 {
            2.0 * (-autocovariance).sqrt()
        } else {
            0.0
        }
    }
    
    // Helper statistical functions
    fn covariance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();
        
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / (x.len() - 1) as f64
    }
    
    fn variance(x: &Array1<f64>) -> f64 {
        x.var(0.0)
    }
    
    fn autocovariance(x: &Array1<f64>, lag: usize) -> f64 {
        if lag >= x.len() {
            return 0.0;
        }
        
        let x1 = x.slice(s![..x.len()-lag]);
        let x2 = x.slice(s![lag..]);
        
        Self::covariance(&x1.to_owned(), &x2.to_owned())
    }
}
```

## **PHASE 7: Model Implementation (Priority: Medium)**

### **7.1 Lightweight Ensemble Methods**

```rust
// Dependencies
smartcore = "0.3"

use smartcore::tree::decision_tree_classifier::*;
use smartcore::ensemble::random_forest_classifier::*;

struct FinancialMLModel {
    base_models: Vec<DecisionTreeClassifier<f64, i32>>,
    model_weights: Array1<f64>,
    feature_importance: Array1<f64>,
    trained: bool,
}

impl FinancialMLModel {
    fn fit_ensemble(
        &mut self,
        features: &Array2<f64>,
        labels: &Array1<i32>,
        sample_weights: &Array1<f64>,
    ) -> Result<ModelMetrics, Box<dyn std::error::Error>> {
        // Bagged ensemble with sample weighting
        let n_estimators = 100;
        let sample_calculator = SampleWeightCalculator::new(/* params */);
        
        for i in 0..n_estimators {
            let bootstrap_indices = sample_calculator.sequential_bootstrap(features.nrows());
            let bootstrap_features = Self::select_rows(features, &bootstrap_indices);
            let bootstrap_labels = Self::select_elements(labels, &bootstrap_indices);
            let bootstrap_weights = Self::select_elements(sample_weights, &bootstrap_indices);
            
            let mut tree = DecisionTreeClassifier::fit(
                &bootstrap_features,
                &bootstrap_labels,
                DecisionTreeClassifierParameters::default(),
            )?;
            
            self.base_models.push(tree);
        }
        
        self.trained = true;
        self.calculate_feature_importance(features, labels);
        
        Ok(ModelMetrics {
            cv_score: self.cross_validate(features, labels)?,
            n_estimators: n_estimators,
            feature_importance: self.feature_importance.clone(),
        })
    }
    
    fn predict_with_confidence(&self, features: &Array1<f64>) -> (i32, f64) {
        if !self.trained {
            return (1, 0.0); // Default to hold
        }
        
        let predictions: Vec<i32> = self.base_models.iter()
            .map(|model| model.predict(&features.insert_axis(Axis(0))).unwrap()[0])
            .collect();
        
        // Majority vote with confidence
        let mut vote_counts = [0; 3]; // [sell, hold, buy]
        for &pred in &predictions {
            vote_counts[pred as usize] += 1;
        }
        
        let (winning_class, max_votes) = vote_counts.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .unwrap();
        
        let confidence = *max_votes as f64 / predictions.len() as f64;
        (winning_class as i32, confidence)
    }
    
    fn calculate_feature_importance(&mut self, features: &Array2<f64>, labels: &Array1<i32>) {
        // Permutation importance calculation
        let baseline_score = self.evaluate_model(features, labels);
        let mut importance = Array1::zeros(features.ncols());
        
        for feature_idx in 0..features.ncols() {
            let mut shuffled_features = features.clone();
            Self::shuffle_column(&mut shuffled_features, feature_idx);
            
            let shuffled_score = self.evaluate_model(&shuffled_features, labels);
            importance[feature_idx] = baseline_score - shuffled_score;
        }
        
        self.feature_importance = importance;
    }
    
    // Helper methods
    fn select_rows(array: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut result = Array2::zeros((indices.len(), array.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&array.row(idx));
        }
        result
    }
    
    fn select_elements(array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        indices.iter().map(|&i| array[i]).collect::<Vec<_>>().into()
    }
    
    fn shuffle_column(array: &mut Array2<f64>, col_idx: usize) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut column: Vec<f64> = array.column(col_idx).to_vec();
        column.shuffle(&mut rng);
        array.column_mut(col_idx).assign(&Array1::from(column));
    }
    
    fn evaluate_model(&self, features: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        // Simple accuracy calculation
        let mut correct = 0;
        let n_samples = features.nrows();
        
        for i in 0..n_samples {
            let (pred, _) = self.predict_with_confidence(&features.row(i).to_owned());
            if pred == labels[i] {
                correct += 1;
            }
        }
        
        correct as f64 / n_samples as f64
    }
    
    fn cross_validate(&self, features: &Array2<f64>, labels: &Array1<i32>) -> Result<f64, Box<dyn std::error::Error>> {
        // Placeholder for cross-validation implementation
        Ok(0.75)
    }
}

#[derive(Debug)]
struct ModelMetrics {
    cv_score: f64,
    n_estimators: usize,
    feature_importance: Array1<f64>,
}
```

## **PHASE 8: Configuration & Error Handling (Priority: Low)**

### **8.1 Clean Configuration**

```rust
// Dependencies
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
anyhow = "1.0"
thiserror = "1.0"

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelConfig {
    // Fractional differentiation
    frac_diff_threshold: f64,
    auto_optimize_d: bool,
    
    // Triple barrier
    profit_multiplier: f64,
    stop_multiplier: f64,
    max_holding_period: usize,
    
    // Cross-validation
    n_splits: usize,
    embargo_percent: f64,
    
    // Model parameters
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    
    // Feature engineering
    volatility_window: usize,
    enable_microstructural_features: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            frac_diff_threshold: 1e-5,
            auto_optimize_d: true,
            profit_multiplier: 2.0,
            stop_multiplier: 1.5,
            max_holding_period: 20,
            n_splits: 3,
            embargo_percent: 0.01,
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 5,
            volatility_window: 20,
            enable_microstructural_features: false,
        }
    }
}

#[derive(thiserror::Error, Debug)]
enum FinancialMLError {
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Information leakage detected in cross-validation")]
    InformationLeakage,
    
    #[error("Invalid barrier configuration: profit_mult={profit}, stop_mult={stop}")]
    InvalidBarrierConfig { profit: f64, stop: f64 },
    
    #[error("Model not trained - call fit() first")]
    ModelNotTrained,
    
    #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
    FeatureDimensionMismatch { expected: usize, actual: usize },
}
```

### **8.2 Usage Example**

```rust
use anyhow::Result;

fn main() -> Result<()> {
    // Load configuration
    let config = ModelConfig::default();
    
    // Load and prepare data
    let prices = FinancialSeries::from_csv("data/prices.csv")?;
    
    // Feature engineering pipeline
    let frac_diff = FractionalDifferentiator::new(0.35, config.frac_diff_threshold);
    let stationary_prices = frac_diff.transform(&prices.values);
    
    let volatility = prices.rolling_volatility(config.volatility_window);
    let returns = prices.pct_change(1);
    
    // Combine features
    let features = Array2::from_shape_fn((prices.len(), 3), |(i, j)| {
        match j {
            0 => stationary_prices[i],
            1 => volatility[i],
            2 => returns[i],
            _ => 0.0,
        }
    });
    
    // Generate labels
    let labeler = TripleBarrierLabeler::new(TripleBarrierConfig {
        profit_mult: config.profit_multiplier,
        stop_mult: config.stop_multiplier,
        max_hold: config.max_holding_period,
        min_return: 0.001,
    });
    
    let events: Vec<Event> = prices.timestamps.iter()
        .enumerate()
        .skip(100) // Skip initial period
        .take(prices.len() - 200) // Leave buffer
        .map(|(i, &timestamp)| Event {
            timestamp,
            t1: Some(timestamp + Duration::days(config.max_holding_period as i64)),
            target: volatility[i],
            side: 1.0, // Long positions
        })
        .collect();
    
    let labels = labeler.generate_labels_parallel(&prices, &events);
    
    // Sample weighting
    let weight_calculator = SampleWeightCalculator::from_events(&events);
    let sample_weights = weight_calculator.calculate_average_uniqueness();
    
    // Cross-validation setup
    let cv = PurgedKFold {
        n_splits: config.n_splits,
        embargo_pct: config.embargo_percent,
        label_endtimes: events.iter().map(|e| e.t1.unwrap()).collect(),
    };
    
    // Train model
    let mut model = FinancialMLModel::new();
    let metrics = model.fit_ensemble(&features, &labels, &sample_weights)?;
    
    println!("Training completed:");
    println!("  CV Score: {:.3}", metrics.cv_score);
    println!("  N Estimators: {}", metrics.n_estimators);
    println!("  Top features: {:?}", 
        metrics.feature_importance.iter()
            .enumerate()
            .collect::<Vec<_>>()
    );
    
    // Make predictions
    let (prediction, confidence) = model.predict_with_confidence(&features.row(features.nrows()-1).to_owned());
    println!("Latest prediction: {} (confidence: {:.3})", 
        match prediction {
            0 => "SELL",
            1 => "HOLD", 
            2 => "BUY",
            _ => "UNKNOWN"
        },
        confidence
    );
    
    Ok(())
}
```

## **PHASE 9: Testing & Validation (Priority: High)**

### **9.1 Unit Tests for Financial Logic**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_fractional_differentiation_d_zero() {
        // d=0 should return original series
        let series = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let differentiator = FractionalDifferentiator::new(0.0, 1e-5);
        let result = differentiator.transform(&series);
        
        // Should be identical after window size
        for i in differentiator.window_size..series.len() {
            assert_relative_eq!(result[i], series[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_fractional_differentiation_d_one() {
        // d=1 should approximate first differences
        let series = Array1::from(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
        let differentiator = FractionalDifferentiator::new(1.0, 1e-5);
        let result = differentiator.transform(&series);
        
        // Should approximate differences for large enough window
        let expected_diffs = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        for i in 1..expected_diffs.len() {
            assert_relative_eq!(result[i+1], expected_diffs[i], epsilon = 0.1);
        }
    }
    
    #[test] 
    fn test_triple_barrier_profit_target_hit() {
        let prices = vec![100.0, 101.0, 102.0, 105.0, 103.0]; // Profit target hit
        let price_series = FinancialSeries::from_values(prices);
        
        let config = TripleBarrierConfig {
            profit_mult: 0.03, // 3% profit target
            stop_mult: 0.02,   // 2% stop loss
            max_hold: 10,
            min_return: 0.001,
        };
        
        let event = Event {
            timestamp: price_series.timestamps[0],
            t1: Some(price_series.timestamps[4]),
            target: 1.0, // Unit volatility
            side: 1.0,   // Long position
        };
        
        let labeler = TripleBarrierLabeler { config };
        let label = labeler.label_single_event(&price_series, &event);
        
        assert_eq!(label, 2); // Should be BUY signal (profit target hit)
    }
    
    #[test]
    fn test_purged_cv_no_overlap() {
        let n_samples = 100;
        let label_endtimes: Vec<OffsetDateTime> = (0..n_samples)
            .map(|i| OffsetDateTime::now_utc() + Duration::days(i as i64))
            .collect();
        
        let cv = PurgedKFold {
            n_splits: 3,
            embargo_pct: 0.05,
            label_endtimes,
        };
        
        let splits = cv.split(n_samples);
        
        // Verify no overlap between train and test sets
        for (train_indices, test_indices) in splits {
            for &test_idx in &test_indices {
                assert!(!train_indices.contains(&test_idx), 
                    "Found overlap: test index {} in training set", test_idx);
            }
        }
    }
    
    #[test]
    fn test_sample_weights_sum_reasonable() {
        let n_samples = 50;
        let label_endtimes: Vec<OffsetDateTime> = (0..n_samples)
            .map(|i| OffsetDateTime::now_utc() + Duration::hours(i as i64))
            .collect();
        
        // Create some overlaps
        let mut overlap_matrix = Array2::eye(n_samples);
        for i in 0..n_samples-1 {
            overlap_matrix[[i, i+1]] = true;
            overlap_matrix[[i+1, i]] = true;
        }
        
        let calculator = SampleWeightCalculator {
            label_endtimes,
            overlap_matrix,
        };
        
        let weights = calculator.calculate_average_uniqueness();
        
        // Weights should be positive and reasonable
        assert!(weights.iter().all(|&w| w > 0.0 && w <= 1.0));
        
        // Average weight should be reasonable (not too concentrated)
        let mean_weight = weights.mean().unwrap();
        assert!(mean_weight > 0.1 && mean_weight < 1.0);
    }
    
    #[test]
    fn test_vpin_calculation() {
        let buy_volumes = vec![100.0, 150.0, 200.0, 120.0];
        let sell_volumes = vec![80.0, 130.0, 180.0, 140.0];
        
        let calculator = VPINCalculator { window_size: 4 };
        let vpin = calculator.calculate(&buy_volumes, &sell_volumes);
        
        // VPIN should be between 0 and 1
        assert!(vpin >= 0.0 && vpin <= 1.0);
        
        // For this data: total_imbalance = |20| + |20| + |20| + |-20| = 80
        // total_volume = 180 + 280 + 380 + 260 = 1100
        // VPIN = 80/1100 ≈ 0.0727
        assert_relative_eq!(vpin, 80.0/1100.0, epsilon = 1e-3);
    }
}
```

### **9.2 Integration Tests**

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_pipeline_synthetic_data() {
        // Generate synthetic data with known patterns
        let n_samples = 1000;
        let prices = generate_synthetic_price_series(n_samples, 0.02); // 2% daily vol
        
        // Run full pipeline
        let config = ModelConfig::default();
        let result = run_full_pipeline(&prices, &config);
        
        assert!(result.is_ok());
        let metrics = result.unwrap();
        
        // Sanity checks
        assert!(metrics.cv_score > 0.3); // Should beat random
        assert!(metrics.cv_score < 1.0); // Should not be perfect
        assert_eq!(metrics.n_estimators, config.n_estimators);
    }
    
    fn generate_synthetic_price_series(n: usize, volatility: f64) -> FinancialSeries {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        
        let mut prices = vec![100.0]; // Start at $100
        let mut timestamps = vec![OffsetDateTime::now_utc()];
        
        for i in 1..n {
            let return_val = normal.sample(&mut rng);
            prices.push(prices[i-1] * (1.0 + return_val));
            timestamps.push(timestamps[i-1] + Duration::days(1));
        }
        
        FinancialSeries::new(timestamps, Array1::from(prices))
    }
    
    fn run_full_pipeline(prices: &FinancialSeries, config: &ModelConfig) -> Result<ModelMetrics> {
        // This would call the main pipeline logic
        todo!("Implement full integration test")
    }
}
```

## **PHASE 10: Performance Optimization (Priority: Low)**

### **10.1 SIMD Optimizations**

```rust
// Dependencies
wide = "0.7"

use wide::*;

impl FractionalDifferentiator {
    fn transform_simd(&self, series: &Array1<f64>) -> Array1<f64> {
        // SIMD-optimized fractional differentiation
        let n = series.len();
        let mut result = Array1::zeros(n);
        
        // Process 4 doubles at a time using SIMD
        for i in self.window_size..n {
            let start = i - self.window_size + 1;
            let mut sum = f64x4::ZERO;
            
            // Vectorized dot product
            for (weight_chunk, value_chunk) in self.weights.chunks_exact(4)
                .zip(series.slice(s![start..=i]).as_slice().unwrap().chunks_exact(4)) {
                
                let w = f64x4::from_array(*weight_chunk.try_into().unwrap());
                let v = f64x4::from_array(*value_chunk.try_into().unwrap());
                sum = sum + w * v;
            }
            
            result[i] = sum.reduce_add();
            
            // Handle remainder
            let remainder_start = (self.weights.len() / 4) * 4;
            for j in remainder_start..self.weights.len() {
                result[i] += self.weights[j] * series[start + j];
            }
        }
        
        result
    }
}
```

### **10.2 Memory Pool Allocation**

```rust
// For high-frequency processing
struct MemoryPool {
    buffers: Vec<Vec<f64>>,
    available: Vec<bool>,
}

impl MemoryPool {
    fn get_buffer(&mut self, size: usize) -> Option<&mut Vec<f64>> {
        for (i, available) in self.available.iter_mut().enumerate() {
            if *available && self.buffers[i].capacity() >= size {
                *available = false;
                self.buffers[i].clear();
                return Some(&mut self.buffers[i]);
            }
        }
        None
    }
    
    fn return_buffer(&mut self, buffer: *mut Vec<f64>) {
        // Mark buffer as available
        // Unsafe pointer arithmetic for zero-copy returns
    }
}
```

## **Build Configuration**

### **Cargo.toml**

```toml
[package]
name = "financial-ml"
version = "0.1.0"
edition = "2021"

[dependencies]
# Data structures
ndarray = { version = "0.15", features = ["approx", "blas"] }
ndarray-stats = "0.5"
ndarray-linalg = "0.16"
polars = { version = "0.35", features = ["lazy", "temporal"] }

# Time handling
time = { version = "0.3", features = ["serde", "parsing", "formatting"] }

# ML algorithms
smartcore = "0.3"

# Parallel processing
rayon = "1.8"

# Performance
wide = "0.7"

# Statistics
statrs = "0.16"

# Serialization
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
bincode = "1.3"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Utilities
derive_more = "0.99"
rand = "0.8"
rand_distr = "0.4"

[dev-dependencies]
approx = "0.5"
criterion = "0.5"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3

[profile.dev]
opt-level = 1

[[bench]]
name = "fractional_diff_bench"
harness = false

[[bench]]
name = "cross_validation_bench" 
harness = false
```

## **Key Implementation Priorities**

1. **Phase 1-3 (Critical)**: Core data structures, fractional differentiation, and triple barrier labeling
2. **Phase 4 (Critical)**: Purged cross-validation to prevent information leakage
3. **Phase 5 (High)**: Sample weighting for non-IID financial data
4. **Phase 7 (Medium)**: Lightweight ensemble models using smartcore
5. **Phase 9 (High)**: Comprehensive testing suite
6. **Phases 6,8,10 (Low)**: Microstructural features, configuration, and optimization

This Rust-first approach eliminates the architectural complexity of the original implementation while maintaining scientific rigor based on López de Prado's methodologies. The use of lightweight, composable libraries provides significant performance benefits without the overhead of unnecessary abstractions.
