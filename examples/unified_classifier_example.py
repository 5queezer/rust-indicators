#!/usr/bin/env python3
"""
Unified Classifier Usage Example

This example demonstrates how to use the UnifiedClassifier which combines
pattern recognition and trading classification in a single, flexible model
that can operate in Pattern, Trading, or Hybrid modes.
"""

import numpy as np
import pandas as pd
from rust_indicators import UnifiedClassifier, ClassifierMode

def generate_unified_data(n_samples=1000):
    """Generate comprehensive data for unified classifier demonstration."""
    np.random.seed(42)
    
    # Generate synthetic price series
    returns = np.random.normal(0.0005, 0.015, n_samples).astype(np.float32)
    prices = (100 * np.exp(np.cumsum(returns))).astype(np.float32)
    
    # Create OHLC data
    opens = (prices * (1 + np.random.normal(0, 0.001, n_samples))).astype(np.float32)
    highs = (np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))).astype(np.float32)
    lows = (np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))).astype(np.float32)
    closes = prices.astype(np.float32)
    
    # Generate trading features (7 features)
    trading_features = np.zeros((n_samples, 7), dtype=np.float32)
    volatility = np.abs(np.random.normal(0.02, 0.01, n_samples)).astype(np.float32)
    
    trading_features[:, 0] = np.random.normal(50, 20, n_samples)  # RSI
    trading_features[:, 1] = np.random.normal(1.0, 0.1, n_samples)  # MA ratio
    trading_features[:, 2] = volatility  # Volatility
    trading_features[:, 3] = np.random.lognormal(0, 0.5, n_samples)  # Volume
    trading_features[:, 4] = np.random.normal(0.5, 0.3, n_samples)  # BB position
    trading_features[:, 5] = np.random.normal(0, 0.1, n_samples)  # MACD
    trading_features[:, 6] = returns / (volatility + 1e-8)  # Normalized returns
    
    # Normalize trading features
    trading_features[:, 0] = np.clip(trading_features[:, 0], 0, 100)
    trading_features[:, 1] = np.clip(trading_features[:, 1], 0.5, 2.0)
    trading_features[:, 2] = np.clip(trading_features[:, 2], 0.005, 0.1)
    trading_features[:, 3] = np.clip(trading_features[:, 3], 0.1, 5.0)
    trading_features[:, 4] = np.clip(trading_features[:, 4], 0, 1)
    trading_features[:, 6] = np.clip(trading_features[:, 6], -5, 5)
    
    # Generate pattern signals (5 patterns)
    pattern_names = ["doji", "hammer", "engulfing", "shooting_star", "spinning_top"]
    n_patterns = len(pattern_names)
    pattern_signals = np.zeros((n_samples, n_patterns), dtype=np.float32)
    
    # Create pattern signals with some predictive power
    future_returns = np.roll(returns, -5)
    for i, pattern in enumerate(pattern_names):
        base_signal = np.random.random(n_samples)
        trend_signal = np.where(future_returns > 0.01, base_signal * 1.5, base_signal * 0.5)
        pattern_signals[:, i] = np.clip(trend_signal, 0, 1)
    
    # Combine features for unified model (trading + pattern features)
    unified_features = np.column_stack([trading_features, pattern_signals])
    
    # Generate labels based on future returns
    labels = np.where(future_returns > 0.015, 2,  # Buy
                     np.where(future_returns < -0.015, 0, 1)).astype(np.int32)  # Sell, Hold
    
    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'trading_features': trading_features,
        'pattern_signals': pattern_signals,
        'unified_features': unified_features,
        'pattern_names': pattern_names,
        'labels': labels,
        'returns': returns,
        'volatility': volatility,
        'prices': prices
    }

def basic_unified_classifier_example():
    """Basic UnifiedClassifier usage in different modes."""
    print("=== Basic Unified Classifier Example ===")
    
    # Generate sample data
    data = generate_unified_data(1000)
    n_features = data['unified_features'].shape[1]
    
    # Initialize unified classifier in hybrid mode
    classifier = UnifiedClassifier(n_features=n_features, mode=ClassifierMode.Hybrid)
    
    print(f"Initialized UnifiedClassifier:")
    print(f"  Features: {n_features}")
    print(f"  Mode: {classifier.get_mode()}")
    
    # Prepare training data
    split_idx = 700
    X_train = data['unified_features'][:split_idx]
    y_train = data['labels'][:split_idx]
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Label distribution: {np.bincount(y_train)}")
    
    # Train in hybrid mode
    print(f"\nTraining in Hybrid mode...")
    results = classifier.train_unified(
        X=X_train,
        y=y_train,
        learning_rate=0.01
    )
    
    print(f"Hybrid Training Results:")
    print(f"  CV Mean: {results['cv_mean']:.4f}")
    print(f"  Pattern Score: {results.get('pattern_score', 'N/A')}")
    print(f"  Trading Score: {results.get('trading_score', 'N/A')}")
    print(f"  Mode: {results['mode']}")
    
    # Get feature importance and mode-specific weights
    importance = classifier.get_feature_importance()
    mode_weights = classifier.get_mode_weights()
    
    feature_names = ['RSI', 'MA_Ratio', 'Volatility', 'Volume', 'BB_Position', 'MACD', 'Norm_Returns'] + data['pattern_names']
    
    print(f"\nTop 5 Features by Importance:")
    top_features = np.argsort(importance)[-5:][::-1]
    for i in top_features:
        print(f"  {feature_names[i]}: {importance[i]:.4f}")
    
    # Test predictions
    X_test = data['unified_features'][split_idx:]
    y_test = data['labels'][split_idx:]
    
    print(f"\nMaking predictions on {len(X_test)} test samples...")
    
    # Single prediction
    sample_idx = 0
    prediction, confidence = classifier.predict_with_confidence(X_test[sample_idx])
    
    print(f"\nSample Prediction (Hybrid mode):")
    print(f"  Prediction: {prediction} ({'Buy' if prediction == 2 else 'Sell' if prediction == 0 else 'Hold'})")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Actual: {y_test[sample_idx]}")
    
    return classifier, data

def mode_switching_example():
    """Example of switching between different classifier modes."""
    print("\n=== Mode Switching Example ===")
    
    data = generate_unified_data(1200)
    n_features = data['unified_features'].shape[1]
    
    # Initialize classifier
    classifier = UnifiedClassifier(n_features=n_features, mode=ClassifierMode.Hybrid)
    
    # Prepare data
    split_idx = 800
    X_train = data['unified_features'][:split_idx]
    y_train = data['labels'][:split_idx]
    X_test = data['unified_features'][split_idx:]
    y_test = data['labels'][split_idx:]
    
    # Test each mode
    modes = [ClassifierMode.Pattern, ClassifierMode.Trading, ClassifierMode.Hybrid]
    mode_results = {}
    
    for mode in modes:
        print(f"\n--- Testing {mode} Mode ---")
        
        # Set mode and train
        classifier.set_mode(mode)
        print(f"Switched to {classifier.get_mode()} mode")
        
        # Train with mode-specific method
        if mode == ClassifierMode.Pattern:
            results = classifier.train_pattern_mode_explicit(X_train, y_train, 0.01)
        elif mode == ClassifierMode.Trading:
            results = classifier.train_trading_mode_explicit(X_train, y_train, 0.01)
        else:  # Hybrid
            results = classifier.train_hybrid_mode_explicit(X_train, y_train, 0.01)
        
        print(f"Training Results:")
        print(f"  CV Mean: {results['cv_mean']:.4f}")
        if 'cv_std' in results:
            print(f"  CV Std: {results['cv_std']:.4f}")
        
        # Test predictions
        correct = 0
        total_confident = 0
        confidence_threshold = 0.3
        
        for i in range(len(X_test)):
            pred, conf = classifier.predict_with_confidence(X_test[i])
            
            if conf > confidence_threshold:
                total_confident += 1
                if pred == y_test[i]:
                    correct += 1
        
        if total_confident > 0:
            accuracy = correct / total_confident
            coverage = total_confident / len(X_test)
            
            mode_results[str(mode)] = {
                'accuracy': accuracy,
                'coverage': coverage,
                'cv_score': results['cv_mean']
            }
            
            print(f"Test Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Coverage: {coverage:.4f}")
    
    # Compare modes
    print(f"\n--- Mode Comparison ---")
    for mode_name, results in mode_results.items():
        print(f"{mode_name}:")
        print(f"  CV Score: {results['cv_score']:.4f}")
        print(f"  Test Accuracy: {results['accuracy']:.4f}")
        print(f"  Coverage: {results['coverage']:.4f}")
    
    return mode_results

def advanced_unified_classifier_example():
    """Advanced usage with custom parameters and analysis."""
    print("\n=== Advanced Unified Classifier Example ===")
    
    # Generate larger dataset
    data = generate_unified_data(2000)
    n_features = data['unified_features'].shape[1]
    
    # Initialize with custom parameters
    classifier = UnifiedClassifier(n_features=n_features, mode=ClassifierMode.Hybrid)
    classifier.set_embargo_pct(0.02)  # 2% embargo
    classifier.set_pattern_duration(8)  # 8-bar pattern duration
    
    print(f"Advanced UnifiedClassifier setup:")
    print(f"  Features: {n_features}")
    print(f"  Mode: {classifier.get_mode()}")
    print(f"  Embargo: 2%")
    print(f"  Pattern Duration: 8 bars")
    
    # Use more training data
    split_idx = 1500
    X_train = data['unified_features'][:split_idx]
    y_train = data['labels'][:split_idx]
    
    # Train with custom learning rate
    print(f"\nTraining on {len(X_train)} samples...")
    results = classifier.train_unified(
        X=X_train,
        y=y_train,
        learning_rate=0.015  # Custom learning rate
    )
    
    print(f"Advanced Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Analyze feature contributions
    importance = classifier.get_feature_importance()
    mode_weights = classifier.get_mode_weights()
    
    # Separate trading and pattern features
    trading_importance = importance[:7]
    pattern_importance = importance[7:]
    
    trading_names = ['RSI', 'MA_Ratio', 'Volatility', 'Volume', 'BB_Position', 'MACD', 'Norm_Returns']
    pattern_names = data['pattern_names']
    
    print(f"\nTop Trading Features:")
    trading_sorted = sorted(zip(trading_names, trading_importance), key=lambda x: x[1], reverse=True)
    for name, imp in trading_sorted[:3]:
        print(f"  {name}: {imp:.4f}")
    
    print(f"\nTop Pattern Features:")
    pattern_sorted = sorted(zip(pattern_names, pattern_importance), key=lambda x: x[1], reverse=True)
    for name, imp in pattern_sorted[:3]:
        print(f"  {name}: {imp:.4f}")
    
    # Test with different confidence thresholds
    X_test = data['unified_features'][split_idx:]
    y_test = data['labels'][split_idx:]
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    print(f"\nPerformance at different confidence thresholds:")
    
    for threshold in thresholds:
        correct = 0
        total_confident = 0
        
        for i in range(len(X_test)):
            pred, conf = classifier.predict_with_confidence(X_test[i])
            
            if conf > threshold:
                total_confident += 1
                if pred == y_test[i]:
                    correct += 1
        
        if total_confident > 0:
            accuracy = correct / total_confident
            coverage = total_confident / len(X_test)
            print(f"  Threshold {threshold}: Accuracy={accuracy:.3f}, Coverage={coverage:.3f}")
    
    return classifier

def label_generation_comparison():
    """Compare different label generation methods."""
    print("\n=== Label Generation Comparison ===")
    
    data = generate_unified_data(800)
    classifier = UnifiedClassifier(n_features=data['unified_features'].shape[1], mode=ClassifierMode.Hybrid)
    
    # Generate triple barrier labels
    print("Generating triple barrier labels...")
    triple_labels = classifier.create_triple_barrier_labels(
        prices=data['prices'],
        volatility=data['volatility'],
        profit_mult=2.0,
        stop_mult=1.5,
        max_hold=20
    )
    
    # Generate pattern-based labels
    print("Generating pattern-based labels...")
    pattern_labels = classifier.create_pattern_labels(
        open_prices=data['opens'],
        high_prices=data['highs'],
        low_prices=data['lows'],
        close_prices=data['closes'],
        future_periods=10,
        profit_threshold=0.02,
        stop_threshold=0.02
    )
    
    # Compare label distributions
    print(f"\nLabel Distribution Comparison:")
    
    triple_dist = np.bincount(triple_labels, minlength=3)
    pattern_dist = np.bincount(pattern_labels, minlength=3)
    
    label_names = ['Sell', 'Hold', 'Buy']
    
    print(f"Triple Barrier Labels:")
    for i, (name, count) in enumerate(zip(label_names, triple_dist)):
        pct = count / len(triple_labels) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    print(f"Pattern-Based Labels:")
    for i, (name, count) in enumerate(zip(label_names, pattern_dist)):
        pct = count / len(pattern_labels) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Calculate agreement between methods
    agreement = np.sum(triple_labels == pattern_labels) / len(triple_labels)
    print(f"\nLabel Agreement: {agreement:.3f}")
    
    return triple_labels, pattern_labels

def cross_validation_comparison():
    """Compare different cross-validation strategies."""
    print("\n=== Cross-Validation Comparison ===")
    
    data = generate_unified_data(1000)
    classifier = UnifiedClassifier(n_features=data['unified_features'].shape[1], mode=ClassifierMode.Hybrid)
    
    n_samples = 800
    n_splits = 3
    
    # Test purged CV
    print("Testing purged cross-validation...")
    purged_splits = classifier.create_purged_cv_splits(n_samples, n_splits, 0.02)
    
    # Test pattern-aware CV
    print("Testing pattern-aware cross-validation...")
    pattern_splits = classifier.create_pattern_aware_cv_splits(n_samples, n_splits, 10)
    
    # Analyze splits
    def analyze_splits(splits, name):
        train_sizes = [len(train) for train, _ in splits]
        test_sizes = [len(test) for _, test in splits]
        
        print(f"\n{name} Results:")
        print(f"  Valid splits: {len(splits)}")
        print(f"  Avg train size: {np.mean(train_sizes):.0f}")
        print(f"  Avg test size: {np.mean(test_sizes):.0f}")
        
        # Check for overlaps
        total_overlap = 0
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            total_overlap += len(overlap)
        
        print(f"  Data leakage: {total_overlap} samples")
        
        return splits
    
    purged_splits = analyze_splits(purged_splits, "Purged CV")
    pattern_splits = analyze_splits(pattern_splits, "Pattern-Aware CV")
    
    return purged_splits, pattern_splits

if __name__ == "__main__":
    # Run examples
    print("Unified Classifier Examples")
    print("=" * 50)
    
    # Basic example
    basic_classifier, data = basic_unified_classifier_example()
    
    # Mode switching example
    mode_results = mode_switching_example()
    
    # Advanced example
    advanced_classifier = advanced_unified_classifier_example()
    
    # Label generation comparison
    triple_labels, pattern_labels = label_generation_comparison()
    
    # Cross-validation comparison
    purged_splits, pattern_splits = cross_validation_comparison()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nKey takeaways:")
    print("- UnifiedClassifier combines pattern and trading approaches")
    print("- Mode switching allows flexible strategy selection")
    print("- Hybrid mode leverages both pattern and trading features")
    print("- Different labeling methods suit different trading styles")
    print("- Cross-validation prevents overfitting in time series data")