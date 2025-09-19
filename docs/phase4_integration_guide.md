# Phase 4 Integration Guide

## Overview

Phase 4 introduces advanced overfitting detection capabilities to the rust_indicators ML ecosystem through CombinatorialPurgedCV and statistical overfitting analysis. This guide demonstrates how to integrate Phase 4 components with existing models while maintaining backward compatibility.

## Key Components

### CombinatorialPurgedCV
- **Purpose**: Enhanced cross-validation using C(N,k) combinations
- **Benefits**: More robust validation than traditional k-fold CV
- **Implementation**: Generates all possible combinations of training/test splits

### OverfittingDetection
- **Purpose**: Statistical analysis of overfitting risk
- **Method**: Probability of Backtest Overfitting (PBO) calculation
- **Output**: Quantitative overfitting risk assessment

## Integration Examples

### 1. PatternClassifier Integration

```rust
use rust_indicators::ml::models::PatternClassifier;
use rust_indicators::ml::components::Phase4Config;

// Create classifier
let mut classifier = PatternClassifier::new(7)?;

// Enable Phase 4 validation
classifier.enable_phase4_validation(0.02, 8, 2, 100, 20)?;

// Train with overfitting detection
let results = classifier.train_with_phase4_validation(X, y, 0.01, true)?;

// Check overfitting analysis
if let Some(analysis) = classifier.get_overfitting_analysis(py)? {
    println!("PBO: {:.3}", analysis["pbo_value"]);
    println!("Overfit Risk: {}", if analysis["is_overfit"] > 0.5 { "HIGH" } else { "LOW" });
}
```

### 2. TradingClassifier Integration

```rust
use rust_indicators::ml::models::TradingClassifier;

// Create classifier
let mut classifier = TradingClassifier::new(7)?;

// Enable Phase 4 validation with custom parameters
classifier.enable_phase4_validation(0.02, 10, 2, 150, 30)?;

// Train with enhanced validation
let results = classifier.train_with_overfitting_detection(X, y, 0.01, true)?;

// Get risk assessment
let risk = classifier.assess_overfitting_risk();
println!("Risk Assessment: {}", risk);
```

### 3. UnifiedClassifier Integration

```rust
use rust_indicators::ml::models::{UnifiedClassifier, ClassifierMode};

// Create unified classifier
let mut classifier = UnifiedClassifier::new(10, Some(ClassifierMode::Hybrid))?;

// Enable Phase 4 validation
classifier.enable_phase4_validation(0.02, 8, 2, 100, 20)?;

// Train in hybrid mode with Phase 4
let results = classifier.train_with_overfitting_detection(X, y, 0.01, true)?;

// Analyze results
if let Some(analysis) = classifier.get_overfitting_analysis(py)? {
    println!("Hybrid Mode PBO: {:.3}", analysis["pbo_value"]);
    println!("Statistical Significance: {:.3}", analysis["statistical_significance"]);
}
```

## Convenience API

### Using Phase4Config Builder

```rust
use rust_indicators::ml::components::{Phase4Config, create_phase4_validator};

// Create custom configuration
let config = Phase4Config::new()
    .embargo_pct(0.03)
    .n_groups(10)
    .test_groups(3)
    .min_train_size(200)
    .min_test_size(50)
    .significance_level(0.01);

// Create validator
let validator = create_phase4_validator(config)?;

// Use validator
let splits = validator.create_splits(1000)?;
let pbo_result = validator.calculate_pbo(&in_sample_scores, &out_sample_scores)?;
```

### Migration from Phase 3

```rust
use rust_indicators::ml::components::{migrate_to_combinatorial_cv, create_legacy_cv_splits};

// Legacy Phase 3 approach
let legacy_splits = create_legacy_cv_splits(1000, 5, 0.02)?;

// Migrate to Phase 4
let purged_cv = PurgedCrossValidator::default();
let combinatorial_cv = migrate_to_combinatorial_cv(&purged_cv, 8, 2);

// Compare approaches
let comparison = ValidationComparison::new(1000, Phase4Config::default())?;
println!("{}", comparison.get_validation_summary());
```

## Performance Characteristics

### Validation Complexity
- **Phase 3**: O(k) where k = number of folds (typically 3-5)
- **Phase 4**: O(C(N,k)) where N = groups, k = test groups (typically 28-56 combinations)

### Memory Usage
- **Additional Memory**: ~10-20% increase for combination storage
- **Computation Time**: 5-10x increase due to more thorough validation
- **Statistical Power**: Significantly improved overfitting detection

### Recommended Settings

| Dataset Size | n_groups | test_groups | embargo_pct | Expected Combinations |
|--------------|----------|-------------|-------------|----------------------|
| < 1000       | 6        | 2           | 0.02        | 15                   |
| 1000-5000    | 8        | 2           | 0.02        | 28                   |
| 5000-10000   | 10       | 2           | 0.02        | 45                   |
| > 10000      | 12       | 3           | 0.01        | 220                  |

## Backward Compatibility

### Existing Code Compatibility
All existing Phase 3 code continues to work without modification:

```rust
// This Phase 3 code still works
let mut classifier = TradingClassifier::new(7)?;
let results = classifier.train_scientific(X, y, 0.01)?;
```

### Gradual Migration
Enable Phase 4 features incrementally:

```rust
// Step 1: Enable Phase 4 validation
classifier.enable_phase4_validation(0.02, 8, 2, 100, 20)?;

// Step 2: Use enhanced training (optional)
let results = classifier.train_with_overfitting_detection(X, y, 0.01, true)?;

// Step 3: Analyze overfitting risk
let analysis = classifier.get_overfitting_analysis(py)?;
```

## Error Handling

### Common Issues and Solutions

1. **Insufficient Sample Size**
   ```rust
   // Error: Not enough samples for combinatorial splits
   // Solution: Reduce n_groups or increase dataset size
   let config = Phase4Config::new().n_groups(6).test_groups(2);
   ```

2. **High Computation Time**
   ```rust
   // Solution: Use fewer combinations for large datasets
   let config = Phase4Config::new().n_groups(8).test_groups(2); // 28 combinations
   ```

3. **Memory Constraints**
   ```rust
   // Solution: Process combinations in batches
   let validator = create_phase4_validator(config)?;
   if !validator.validate_config(n_samples) {
       // Adjust configuration
   }
   ```

## Best Practices

### 1. Configuration Selection
- Start with default settings for most use cases
- Increase n_groups for larger datasets (>5000 samples)
- Use test_groups=2 for most applications
- Adjust embargo_pct based on data frequency (higher for high-frequency data)

### 2. Performance Optimization
- Use Phase 4 validation during model development
- Consider traditional CV for production if speed is critical
- Cache validation results when possible

### 3. Interpretation Guidelines
- **PBO < 0.3**: Low overfitting risk, good generalization expected
- **PBO 0.3-0.6**: Moderate risk, review model complexity
- **PBO > 0.6**: High risk, likely overfitted model
- **PBO > 0.8**: Critical risk, model needs significant revision

### 4. Integration Strategy
1. **Development Phase**: Use Phase 4 for thorough validation
2. **Testing Phase**: Compare Phase 3 vs Phase 4 results
3. **Production Phase**: Choose based on performance requirements
4. **Monitoring Phase**: Periodically re-validate with Phase 4

## Python Integration

All Phase 4 components are fully compatible with Python through PyO3:

```python
from rust_indicators import PatternClassifier
import numpy as np

# Create classifier
classifier = PatternClassifier(n_features=7)

# Enable Phase 4 validation
classifier.enable_phase4_validation(
    embargo_pct=0.02,
    n_groups=8,
    test_groups=2,
    min_train_size=100,
    min_test_size=20
)

# Train with overfitting detection
results = classifier.train_with_phase4_validation(X, y, 0.01, True)

# Analyze results
analysis = classifier.get_overfitting_analysis()
if analysis:
    print(f"PBO: {analysis['pbo_value']:.3f}")
    print(f"Overfit Risk: {'HIGH' if analysis['is_overfit'] else 'LOW'}")
```

## Conclusion

Phase 4 integration provides powerful overfitting detection capabilities while maintaining full backward compatibility. The enhanced validation methods offer more robust model evaluation at the cost of increased computation time. Use the convenience APIs and configuration builders to easily integrate Phase 4 into existing workflows.