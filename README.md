# Rust Indicators

*High-performance technical analysis indicators for quantitative trading with adaptive GPU acceleration*

## Overview

This library balances rigorous performance with elegant simplicity. It adapts to computational needs naturally—using CPU for small datasets and GPU for large ones, without forcing rigid choices.

## Features

### Core Indicators

- **RSI**: Relative Strength Index with optimized momentum calculation
- **EMA/SMA**: Moving averages with ultra-fast implementations  
- **Bollinger Bands**: Statistical volatility bands
- **ATR**: Average True Range for volatility measurement
- **Williams %R**: Momentum oscillator
- **CCI**: Commodity Channel Index
- **VPIN**: Volume-synchronized Probability of Informed Trading

### Advanced Indicators

- **SuperSmoother**: Ehlers low-pass filter with minimal lag
- **Hilbert Transform**: Complex signal analysis for instantaneous amplitude/phase

### Adaptive Architecture

- **CPU Backend**: Optimized sequential computation
- **GPU Backend**: CUDA/WebGPU acceleration for large datasets
- **Adaptive Backend**: Intelligent selection based on workload characteristics

## Installation

```bash
# Basic installation
pip install rust_indicators

# With GPU support (WebGPU)
pip install rust_indicators[gpu]

# With CUDA support
pip install rust_indicators[cuda]

# Development environment
pip install rust_indicators[dev,analysis]
```

## Quick Start

```python
import numpy as np
from rust_indicators import RustTA

# Initialize with automatic backend selection
ta = RustTA()
print(f"Using backend: {ta.device()}")  # "cpu", "gpu", or "adaptive"

# Calculate indicators
prices = np.random.randn(1000).cumsum()
rsi = ta.rsi(prices, period=14)
ema = ta.ema(prices, period=20)

# Advanced indicators
buy_vol = np.random.rand(1000)
sell_vol = np.random.rand(1000)
vpin = ta.vpin(buy_vol, sell_vol, window=50)

# Ehlers indicators
smoothed = ta.supersmoother(prices, period=10)
real, imag = ta.hilbert_transform(prices, lp_period=14)
```

## Performance

- **10-100x faster** than pure Python implementations
- **Automatic GPU acceleration** for datasets > 1500 points (VPIN)
- **Zero-copy NumPy integration** for minimal overhead
- **Adaptive thresholds** calibrated through benchmarking

## Backend Selection

The library automatically selects the optimal backend:

```bash
# Force specific backend
export RUST_INDICATORS_DEVICE=cpu      # CPU only
export RUST_INDICATORS_DEVICE=gpu      # GPU preferred
export RUST_INDICATORS_DEVICE=adaptive # Intelligent selection (default)
```

## Architecture

```
rust_indicators/
├── core/           # Trait definitions and interfaces
├── backends/       # CPU, GPU, and Adaptive implementations  
├── indicators/     # Technical analysis indicators
├── features/       # Feature engineering utilities
├── ml/             # Lightweight ML models
└── utils/          # Benchmarking and testing utilities
```

## Development

```bash
# Build from source
git clone <repository>
cd rust_indicators

# Install development dependencies
pip install -e .[dev]

# Run tests
cargo test
pytest

# Benchmark performance
cargo run --bin simple_vpin_benchmark
```

## Feature Engineering

```python
from rust_indicators import RustFeatures

features = RustFeatures()

# Financial feature engineering
log_returns = features.log_returns(prices)
volatility = features.rolling_volatility(log_returns, window=20)
regime = features.volatility_regime(volatility, threshold=0.02)

# Alternative data structures
volume_bars = features.volume_bars(prices, volumes, threshold=1000)
```

## Machine Learning

```python
from rust_indicators import RustMLModel

# Lightweight model for meta-labeling
model = RustMLModel()
features = np.random.randn(100, 7)  # 7 features per sample
predictions = model.predict(features)
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome. Focus on changes that flow naturally with the existing architecture, avoiding unnecessary complexity.
