#!/usr/bin/env python3
"""
Trading Classifier Usage Example

This example demonstrates how to use the TradingClassifier for scientific trading
signal classification with purged cross-validation and triple barrier labeling.
"""

import numpy as np
from rust_indicators import TradingClassifier

def generate_trading_data(n_samples=1000):
    """Generate sample trading features and price data for demonstration."""
    np.random.seed(42)

    # Generate synthetic price series
    returns = np.random.normal(0.0005, 0.015, n_samples).astype(np.float32)
    prices = (100 * np.exp(np.cumsum(returns))).astype(np.float32)

    # Generate trading features (technical indicators)
    features = np.zeros((n_samples, 7), dtype=np.float32)

    # Feature 1: RSI-like momentum indicator
    features[:, 0] = np.random.normal(50, 20, n_samples)  # RSI values

    # Feature 2: Moving average ratio
    features[:, 1] = np.random.normal(1.0, 0.1, n_samples)  # Price/MA ratio

    # Feature 3: Volatility measure
    volatility = np.abs(np.random.normal(0.02, 0.01, n_samples)).astype(np.float32)
    features[:, 2] = volatility

    # Feature 4: Volume indicator
    features[:, 3] = np.random.lognormal(0, 0.5, n_samples)  # Volume ratio

    # Feature 5: Bollinger Band position
    features[:, 4] = np.random.normal(0.5, 0.3, n_samples)  # BB position

    # Feature 6: MACD-like signal
    features[:, 5] = np.random.normal(0, 0.1, n_samples)  # MACD histogram

    # Feature 7: ATR-normalized price change
    features[:, 6] = returns / (volatility + 1e-8)  # Normalized returns

    # Normalize features to reasonable ranges
    features[:, 0] = np.clip(features[:, 0], 0, 100)  # RSI
    features[:, 1] = np.clip(features[:, 1], 0.5, 2.0)  # MA ratio
    features[:, 2] = np.clip(features[:, 2], 0.005, 0.1)  # Volatility
    features[:, 3] = np.clip(features[:, 3], 0.1, 5.0)  # Volume
    features[:, 4] = np.clip(features[:, 4], 0, 1)  # BB position
    features[:, 6] = np.clip(features[:, 6], -5, 5)  # Normalized returns

    return {
        'features': features,
        'prices': prices,
        'returns': returns,
        'volatility': volatility
    }

def basic_trading_classifier_example():
    """Basic TradingClassifier usage example."""
    print("=== Basic Trading Classifier Example ===")

    # Generate sample data
    data = generate_trading_data(1000)
    n_features = data['features'].shape[1]

    # Initialize trading classifier
    classifier = TradingClassifier(n_features=n_features)

    print(f"Initialized TradingClassifier with {n_features} features")
    print(f"Default embargo percentage: {classifier.get_embargo_pct()}")

    # Prepare training data
    split_idx = 700
    X_train = data['features'][:split_idx]
    returns_train = data['returns'][:split_idx]

    # Generate labels using triple barrier method
    print("\nGenerating triple barrier labels...")
    labels = classifier.create_triple_barrier_labels(
        prices=data['prices'][:split_idx],
        volatility=data['volatility'][:split_idx],
        profit_mult=2.0,
        stop_mult=1.5,
        max_hold=20
    )

    # Analyze label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['Sell', 'Hold', 'Buy']
    print("Label Distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

    # Calculate sample weights based on volatility
    print("\nCalculating volatility-based sample weights...")
    classifier.calculate_sample_weights(returns_train)

    # Train the model
    print(f"\nTraining on {len(X_train)} samples...")
    results = classifier.train_scientific(
        X=X_train,
        y=labels,
        learning_rate=0.01
    )

    print("Training Results:")
    print(f"  CV Mean: {results['cv_mean']:.4f}")
    print(f"  CV Std:  {results['cv_std']:.4f}")
    print(f"  Folds:   {results['n_folds']}")

    # Get feature importance
    importance = classifier.get_feature_importance()
    feature_names = ['RSI', 'MA_Ratio', 'Volatility', 'Volume', 'BB_Position', 'MACD', 'Norm_Returns']

    print("\nFeature Importance:")
    for _i, (name, imp) in enumerate(zip(feature_names, importance)):
        print(f"  {name}: {imp:.4f}")

    # Make predictions on test data
    X_test = data['features'][split_idx:]
    test_prices = data['prices'][split_idx:]
    test_volatility = data['volatility'][split_idx:]

    # Generate test labels
    test_labels = classifier.create_triple_barrier_labels(
        prices=test_prices,
        volatility=test_volatility,
        profit_mult=2.0,
        stop_mult=1.5,
        max_hold=20
    )

    print(f"\nMaking predictions on {len(X_test)} test samples...")

    # Single prediction example
    sample_idx = 0
    prediction, confidence = classifier.predict_with_confidence(X_test[sample_idx])

    print("\nSample Prediction:")
    print(f"  Features: {X_test[sample_idx]}")
    print(f"  Prediction: {prediction} ({'Buy' if prediction == 2 else 'Sell' if prediction == 0 else 'Hold'})")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Actual: {test_labels[sample_idx]}")

    # Batch evaluation
    correct = 0
    total_confident = 0
    confidence_threshold = 0.3

    for i in range(len(X_test)):
        pred, conf = classifier.predict_with_confidence(X_test[i])

        if conf > confidence_threshold:
            total_confident += 1
            if pred == test_labels[i]:
                correct += 1

    if total_confident > 0:
        accuracy = correct / total_confident
        coverage = total_confident / len(X_test)
        print(f"\nTest Results (confidence > {confidence_threshold}):")
        print(f"  Confident predictions: {total_confident}/{len(X_test)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Coverage: {coverage:.4f}")

    return classifier

def advanced_trading_classifier_example():
    """Advanced TradingClassifier usage with custom parameters."""
    print("\n=== Advanced Trading Classifier Example ===")

    # Generate larger dataset
    data = generate_trading_data(2000)
    n_features = data['features'].shape[1]

    # Initialize with custom settings
    classifier = TradingClassifier(n_features=n_features)
    classifier.set_embargo_pct(0.02)  # 2% embargo period

    print("Advanced TradingClassifier setup:")
    print(f"  Features: {n_features}")
    print(f"  Embargo: {classifier.get_embargo_pct()}")

    # Use more training data
    split_idx = 1500
    X_train = data['features'][:split_idx]
    returns_train = data['returns'][:split_idx]

    # Generate labels with different parameters
    print("\nGenerating labels with custom triple barrier parameters...")
    labels = classifier.create_triple_barrier_labels(
        prices=data['prices'][:split_idx],
        volatility=data['volatility'][:split_idx],
        profit_mult=1.8,  # Tighter profit target
        stop_mult=1.2,    # Tighter stop loss
        max_hold=15       # Shorter holding period
    )

    # Calculate sample weights
    classifier.calculate_sample_weights(returns_train)

    # Create custom purged CV splits
    print("Creating purged cross-validation splits...")
    classifier.create_purged_cv_splits(n_samples=len(X_train), n_splits=5, embargo_pct=classifier.get_embargo_pct())

    # Train with higher learning rate
    print("\nTraining with advanced parameters...")
    results = classifier.train_scientific(
        X=X_train,
        y=labels,
        learning_rate=0.02  # Higher learning rate
    )

    print("Advanced Training Results:")
    print(f"  CV Mean: {results['cv_mean']:.4f}")
    print(f"  CV Std:  {results['cv_std']:.4f}")
    print(f"  Folds:   {results['n_folds']}")

    # Analyze feature importance
    importance = classifier.get_feature_importance()
    feature_names = ['RSI', 'MA_Ratio', 'Volatility', 'Volume', 'BB_Position', 'MACD', 'Norm_Returns']

    # Sort features by importance
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Features by Importance:")
    for name, imp in feature_importance[:5]:
        print(f"  {name}: {imp:.4f}")

    # Test with different confidence thresholds
    X_test = data['features'][split_idx:]
    test_labels = classifier.create_triple_barrier_labels(
        prices=data['prices'][split_idx:],
        volatility=data['volatility'][split_idx:],
        profit_mult=1.8,
        stop_mult=1.2,
        max_hold=15
    )

    thresholds = [0.2, 0.3, 0.4, 0.5]
    print("\nPerformance at different confidence thresholds:")

    for threshold in thresholds:
        correct = 0
        total_confident = 0

        for i in range(len(X_test)):
            pred, conf = classifier.predict_with_confidence(X_test[i])

            if conf > threshold:
                total_confident += 1
                if pred == test_labels[i]:
                    correct += 1

        if total_confident > 0:
            accuracy = correct / total_confident
            coverage = total_confident / len(X_test)
            print(f"  Threshold {threshold}: Accuracy={accuracy:.3f}, Coverage={coverage:.3f}")

    return classifier

def sample_weighting_example():
    """Example of sample weighting strategies."""
    print("\n=== Sample Weighting Example ===")

    data = generate_trading_data(800)
    classifier = TradingClassifier(n_features=data['features'].shape[1])

    # Calculate different types of sample weights
    returns = data['returns']

    print("Calculating volatility-based sample weights...")

    # Calculate and store weights internally for training
    classifier.calculate_sample_weights(returns)

    # Manually calculate weights for analysis using the same algorithm as the Rust code
    def calculate_volatility_weights_manual(returns, window_size=20, min_weight=0.1, max_weight=3.0):
        """Manual implementation of volatility weighting for analysis"""
        n = len(returns)
        weights = np.ones(n, dtype=np.float32)
        window = min(window_size, n)

        for i in range(window, n):
            window_start = max(0, i - window)
            window_rets = returns[window_start:i]
            abs_ret = abs(returns[i])
            avg_abs_ret = np.mean(np.abs(window_rets))

            if avg_abs_ret > 0.0:
                weights[i] = np.clip(abs_ret / avg_abs_ret, min_weight, max_weight)

        return weights

    # Calculate weights for analysis
    weights = calculate_volatility_weights_manual(returns)

    # Analyze weight distribution
    print("Sample Weight Statistics:")
    print(f"  Mean: {np.mean(weights):.4f}")
    print(f"  Std:  {np.std(weights):.4f}")
    print(f"  Min:  {np.min(weights):.4f}")
    print(f"  Max:  {np.max(weights):.4f}")

    # Show relationship between returns and weights
    high_return_indices = np.where(np.abs(returns) > 0.02)[0]
    if len(high_return_indices) > 0:
        high_return_weights = weights[high_return_indices]
        normal_weights = np.delete(weights, high_return_indices)

        print("\nWeight Analysis:")
        print(f"  High volatility periods (|return| > 2%): {len(high_return_indices)}")
        print(f"  Average weight (high vol): {np.mean(high_return_weights):.4f}")
        print(f"  Average weight (normal):   {np.mean(normal_weights):.4f}")

    return weights

def cross_validation_example():
    """Example of purged cross-validation."""
    print("\n=== Cross-Validation Example ===")

    data = generate_trading_data(1200)
    classifier = TradingClassifier(n_features=data['features'].shape[1])

    # Test different embargo percentages
    embargo_percentages = [0.01, 0.02, 0.05]
    n_samples = 1000
    n_splits = 3

    print("Testing purged CV with different embargo percentages:")
    print(f"Samples: {n_samples}, Splits: {n_splits}")

    for embargo_pct in embargo_percentages:
        classifier.set_embargo_pct(embargo_pct)

        # Create CV splits
        splits = classifier.create_purged_cv_splits(n_samples, n_splits, embargo_pct)

        # Analyze splits
        train_sizes = [len(train) for train, _ in splits]
        test_sizes = [len(test) for _, test in splits]

        print(f"\nEmbargo {embargo_pct*100:.0f}%:")
        print(f"  Valid splits: {len(splits)}")
        print(f"  Avg train size: {np.mean(train_sizes):.0f}")
        print(f"  Avg test size:  {np.mean(test_sizes):.0f}")

        # Check for data leakage
        total_overlap = 0
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            total_overlap += len(overlap)

        print(f"  Data leakage: {total_overlap} samples")

    return splits

if __name__ == "__main__":
    # Run examples
    print("Trading Classifier Examples")
    print("=" * 50)

    # Basic example
    basic_classifier = basic_trading_classifier_example()

    # Advanced example
    advanced_classifier = advanced_trading_classifier_example()

    # Sample weighting example
    sample_weights = sample_weighting_example()

    # Cross-validation example
    cv_splits = cross_validation_example()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nKey takeaways:")
    print("- TradingClassifier uses scientific methods for trading signals")
    print("- Triple barrier labeling creates realistic trading targets")
    print("- Purged cross-validation prevents data leakage in time series")
    print("- Sample weighting emphasizes high-volatility periods")
    print("- Feature importance identifies most predictive indicators")
