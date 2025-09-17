#!/usr/bin/env python3
"""
Pattern Classifier Usage Example

This example demonstrates how to use the PatternClassifier for pattern recognition
and ensemble trading signals using candlestick patterns and technical indicators.
"""

import numpy as np
from rust_indicators import PatternClassifier

def generate_sample_data(n_samples=1000):
    """Generate sample OHLC data and pattern signals for demonstration."""
    np.random.seed(42)

    # Generate synthetic price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLC data with realistic spreads
    opens = prices * (1 + np.random.normal(0, 0.001, n_samples))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    closes = prices

    # Generate pattern signals (simulated pattern detection results)
    pattern_names = ["doji", "hammer", "engulfing", "shooting_star", "spinning_top"]
    n_patterns = len(pattern_names)

    # Create pattern signals with some correlation to future returns
    future_returns = np.roll(returns, -5)  # 5-period forward returns
    pattern_signals = np.zeros((n_samples, n_patterns))

    for i, _pattern in enumerate(pattern_names):
        # Create pattern signals with different sensitivities
        base_signal = np.random.random(n_samples)
        trend_signal = np.where(future_returns > 0.01, base_signal * 1.5, base_signal * 0.5)
        pattern_signals[:, i] = np.clip(trend_signal, 0, 1)

    # Generate labels based on future returns
    labels = np.where(future_returns > 0.015, 2,  # Buy
                     np.where(future_returns < -0.015, 0, 1))  # Sell, Hold

    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'pattern_signals': pattern_signals,
        'pattern_names': pattern_names,
        'labels': labels
    }

def basic_pattern_classifier_example():
    """Basic PatternClassifier usage example."""
    print("=== Basic Pattern Classifier Example ===")

    # Generate sample data
    data = generate_sample_data(1000)

    # Initialize pattern classifier
    pattern_names = data['pattern_names']
    classifier = PatternClassifier(pattern_names=pattern_names)

    print(f"Initialized PatternClassifier with {len(pattern_names)} patterns:")
    print(f"Patterns: {pattern_names}")

    # Prepare training data
    pattern_features = data['pattern_signals'][:800].astype(np.float32)  # First 800 samples for training
    price_features = np.column_stack([
        data['opens'][:800],
        data['highs'][:800],
        data['lows'][:800],
        data['closes'][:800]
    ]).astype(np.float32)
    labels = data['labels'][:800].astype(np.int32)

    print("\nTraining data shape:")
    print(f"Pattern features: {pattern_features.shape}")
    print(f"Price features: {price_features.shape}")
    print(f"Labels: {labels.shape}")

    # Train the model
    print("\nTraining pattern ensemble...")
    results = classifier.train_pattern_ensemble(
        pattern_features=pattern_features,
        price_features=price_features,
        y=labels,
        pattern_names=pattern_names,
    )

    print("Training Results:")
    print(f"  CV Mean: {results['cv_mean']:.4f}")
    print(f"  CV Std:  {results['cv_std']:.4f}")
    print(f"  Patterns: {results['n_patterns']}")

    # Get pattern importance
    importance = classifier.get_pattern_importance()
    print("\nPattern Importance:")
    for i, pattern in enumerate(pattern_names):
        print(f"  {pattern}: {importance[i]:.4f}")

    # Make predictions on test data
    test_pattern_features = data['pattern_signals'][800:].astype(np.float32)
    test_price_features = np.column_stack([
        data['opens'][800:],
        data['highs'][800:],
        data['lows'][800:],
        data['closes'][800:]
    ]).astype(np.float32)
    test_labels = data['labels'][800:].astype(np.int32)

    print(f"\nMaking predictions on {len(test_labels)} test samples...")

    # Single prediction example
    sample_idx = 0
    prediction, confidence, contributions = classifier.predict_pattern_ensemble(
        pattern_features=test_pattern_features[sample_idx],
        _price_features=test_price_features[sample_idx]
    )

    print("\nSample Prediction:")
    print(f"  Prediction: {prediction} ({'Buy' if prediction == 2 else 'Sell' if prediction == 0 else 'Hold'})")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Actual: {test_labels[sample_idx]}")

    print("\nPattern Contributions:")
    for i, pattern in enumerate(pattern_names):
        print(f"  {pattern}: {contributions[i]:.4f}")

    # Batch predictions
    correct = 0
    total_confident = 0
    confidence_threshold = 0.6

    for i in range(min(50, len(test_labels))):  # Test first 50 samples
        pred, conf, _ = classifier.predict_pattern_ensemble(
            pattern_features=test_pattern_features[i],
            _price_features=test_price_features[i]
        )

        if conf > confidence_threshold:
            total_confident += 1
            if pred == test_labels[i]:
                correct += 1

    if total_confident > 0:
        accuracy = correct / total_confident
        print(f"\nTest Results (confidence > {confidence_threshold}):")
        print(f"  Confident predictions: {total_confident}/50")
        print(f"  Accuracy: {accuracy:.4f}")

    return classifier

def advanced_pattern_classifier_example():
    """Advanced PatternClassifier usage with custom settings."""
    print("\n=== Advanced Pattern Classifier Example ===")

    # Generate larger dataset
    data = generate_sample_data(2000)

    # Initialize with custom confidence threshold
    classifier = PatternClassifier(pattern_names=data['pattern_names'])
    classifier.set_confidence_threshold(0.7)  # Higher confidence threshold

    print("Set confidence threshold to: 0.7")

    # Use more training data
    split_idx = 1500
    pattern_features = data['pattern_signals'][:split_idx].astype(np.float32)
    price_features = np.column_stack([
        data['opens'][:split_idx],
        data['highs'][:split_idx],
        data['lows'][:split_idx],
        data['closes'][:split_idx]
    ]).astype(np.float32)
    labels = data['labels'][:split_idx].astype(np.int32)

    # Train with ensemble method
    print(f"\nTraining on {split_idx} samples...")
    results = classifier.train_pattern_ensemble(
        pattern_features=pattern_features,
        price_features=price_features,
        y=labels,
        pattern_names=data['pattern_names']
    )

    print("Advanced Training Results:")
    print(f"  CV Mean: {results['cv_mean']:.4f}")
    print(f"  CV Std:  {results['cv_std']:.4f}")

    # Analyze pattern performance
    importance = classifier.get_pattern_importance()
    pattern_performance = list(zip(data['pattern_names'], importance))
    pattern_performance.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Performing Patterns:")
    for pattern, score in pattern_performance[:3]:
        print(f"  {pattern}: {score:.4f}")

    # Test on remaining data
    test_pattern_features = data['pattern_signals'][split_idx:].astype(np.float32)
    test_price_features = np.column_stack([
        data['opens'][split_idx:],
        data['highs'][split_idx:],
        data['lows'][split_idx:],
        data['closes'][split_idx:]
    ]).astype(np.float32)
    test_labels = data['labels'][split_idx:].astype(np.int32)

    # Evaluate with different confidence thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]
    print("\nPerformance at different confidence thresholds:")

    for threshold in thresholds:
        classifier.set_confidence_threshold(threshold)
        correct = 0
        total_confident = 0

        for i in range(len(test_labels)):
            pred, conf, _ = classifier.predict_pattern_ensemble(
                pattern_features=test_pattern_features[i],
                _price_features=test_price_features[i]
            )

            if conf > threshold:
                total_confident += 1
                if pred == test_labels[i]:
                    correct += 1

        if total_confident > 0:
            accuracy = correct / total_confident
            coverage = total_confident / len(test_labels)
            print(f"  Threshold {threshold}: Accuracy={accuracy:.3f}, Coverage={coverage:.3f}")

    return classifier

def pattern_label_generation_example():
    """Example of using pattern-based label generation."""
    print("\n=== Pattern Label Generation Example ===")

    data = generate_sample_data(500)
    PatternClassifier(pattern_names=data['pattern_names'])

    # Note: create_pattern_labels is not exposed as a public method
    # This is just a placeholder example
    print("Pattern label generation would be done here...")
    print("(create_pattern_labels method not available in public API)")

    # Create dummy labels for demonstration
    labels = np.random.choice([0, 1, 2], 500)

    # Analyze label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['Sell', 'Hold', 'Buy']

    print("Label Distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

    return labels

if __name__ == "__main__":
    # Run examples
    print("Pattern Classifier Examples")
    print("=" * 50)

    # Basic example
    basic_classifier = basic_pattern_classifier_example()

    # Advanced example
    advanced_classifier = advanced_pattern_classifier_example()

    # Label generation example
    pattern_labels = pattern_label_generation_example()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nKey takeaways:")
    print("- PatternClassifier excels at ensemble pattern recognition")
    print("- Confidence thresholds control prediction quality vs coverage")
    print("- Pattern importance helps identify most predictive patterns")
    print("- Pattern-based labeling creates training targets from OHLC data")
