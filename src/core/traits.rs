//! Core traits for technical indicator backends
//!
//! This module defines the fundamental traits that all indicator backends must implement.
//! The trait system enables polymorphic backend selection and consistent API across
//! different computation strategies (CPU, GPU, Adaptive).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Core trait for technical indicator computation backends
///
/// This trait defines the interface that all indicator backends must implement.
/// It provides a consistent API for computing various technical indicators while
/// allowing different backends to optimize computation strategies.
///
/// # Backend Implementations
///
/// - **CpuBackend**: Pure CPU implementation optimized for sequential computation
/// - **PartialGpuBackend**: GPU-accelerated implementation for supported indicators
/// - **AdaptiveBackend**: Intelligent backend that selects CPU or GPU based on workload
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync + 'static` to support:
/// - Multi-threaded access from Python
/// - Safe sharing across thread boundaries
/// - Static lifetime for global backend instances
///
/// # Error Handling
///
/// All methods return `PyResult<T>` to properly propagate Python exceptions.
/// Implementations should handle computation errors gracefully and provide
/// meaningful error messages for debugging.
///
/// # Example Implementation
///
/// ```rust
/// use rust_indicators::core::traits::IndicatorsBackend;
/// use numpy::{PyArray1, PyReadonlyArray1};
/// use pyo3::prelude::*;
///
/// struct MyBackend;
///
/// impl IndicatorsBackend for MyBackend {
///     fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize)
///         -> PyResult<Py<PyArray1<f64>>> {
///         // Implementation here
///         todo!("Implement RSI calculation")
///     }
///
///     // ... implement other required methods
/// #   fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// #   fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// #   fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> { todo!() }
/// #   fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// #   fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// #   fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// #   fn vpin<'py>(&self, py: Python<'py>, buy_volumes: PyReadonlyArray1<'py, f64>, sell_volumes: PyReadonlyArray1<'py, f64>, window: usize) -> PyResult<Py<PyArray1<f64>>> { todo!() }
/// }
/// ```
pub trait IndicatorsBackend: Send + Sync + 'static {
    /// Calculate Relative Strength Index (RSI)
    ///
    /// RSI is a momentum oscillator that measures the speed and change of price movements.
    /// It oscillates between 0 and 100, with values above 70 typically indicating overbought
    /// conditions and values below 30 indicating oversold conditions.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `prices`: Array of price values (typically closing prices)
    /// - `period`: Number of periods to use for RSI calculation (commonly 14)
    ///
    /// # Returns
    /// `PyArray1<f64>` containing RSI values, with NaN for initial periods where calculation is not possible
    ///
    /// # Formula
    /// RSI = 100 - (100 / (1 + RS))
    /// where RS = Average Gain / Average Loss over the specified period
    fn rsi<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Exponential Moving Average (EMA)
    ///
    /// EMA gives more weight to recent prices, making it more responsive to new information
    /// compared to Simple Moving Average. It's widely used in trend analysis and signal generation.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `prices`: Array of price values
    /// - `period`: Number of periods for EMA calculation
    ///
    /// # Returns
    /// `PyArray1<f64>` containing EMA values
    ///
    /// # Formula
    /// EMA = (Price × Multiplier) + (Previous EMA × (1 - Multiplier))
    /// where Multiplier = 2 / (period + 1)
    fn ema<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Simple Moving Average (SMA)
    ///
    /// SMA is the arithmetic mean of prices over a specified number of periods.
    /// It's a fundamental indicator used for trend identification and smoothing price data.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `values`: Array of values to average
    /// - `period`: Number of periods to include in each average
    ///
    /// # Returns
    /// `PyArray1<f64>` containing SMA values, with NaN for initial periods
    ///
    /// # Formula
    /// SMA = (Sum of prices over period) / period
    fn sma<'py>(&self, py: Python<'py>, values: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Bollinger Bands
    ///
    /// Bollinger Bands consist of a middle band (SMA) and two outer bands that are
    /// standard deviations away from the middle band. They help identify overbought
    /// and oversold conditions relative to recent price action.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `prices`: Array of price values
    /// - `period`: Number of periods for the moving average
    /// - `std_dev`: Number of standard deviations for the bands (typically 2.0)
    ///
    /// # Returns
    /// Tuple of (upper_band, middle_band, lower_band) as `PyArray1<f64>`
    ///
    /// # Formula
    /// - Middle Band = SMA(period)
    /// - Upper Band = Middle Band + (std_dev × Standard Deviation)
    /// - Lower Band = Middle Band - (std_dev × Standard Deviation)
    fn bollinger_bands<'py>(&self, py: Python<'py>, prices: PyReadonlyArray1<'py, f64>, period: usize, std_dev: f64)
        -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)>;

    /// Calculate Average True Range (ATR)
    ///
    /// ATR measures market volatility by calculating the average of true ranges over
    /// a specified period. It's useful for setting stop-loss levels and position sizing.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `high`: Array of high prices
    /// - `low`: Array of low prices
    /// - `close`: Array of closing prices
    /// - `period`: Number of periods for ATR calculation (commonly 14)
    ///
    /// # Returns
    /// `PyArray1<f64>` containing ATR values
    ///
    /// # Formula
    /// True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
    /// ATR = EMA of True Range over the specified period
    fn atr<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Williams %R
    ///
    /// Williams %R is a momentum indicator that measures overbought and oversold levels.
    /// It oscillates between 0 and -100, with values above -20 indicating overbought
    /// conditions and values below -80 indicating oversold conditions.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `high`: Array of high prices
    /// - `low`: Array of low prices
    /// - `close`: Array of closing prices
    /// - `period`: Number of periods for calculation (commonly 14)
    ///
    /// # Returns
    /// `PyArray1<f64>` containing Williams %R values
    ///
    /// # Formula
    /// %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
    fn williams_r<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Commodity Channel Index (CCI)
    ///
    /// CCI measures the variation of a security's price from its statistical mean.
    /// It's used to identify cyclical trends and overbought/oversold conditions.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `high`: Array of high prices
    /// - `low`: Array of low prices
    /// - `close`: Array of closing prices
    /// - `period`: Number of periods for calculation (commonly 20)
    ///
    /// # Returns
    /// `PyArray1<f64>` containing CCI values
    ///
    /// # Formula
    /// CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)
    /// where Typical Price = (High + Low + Close) / 3
    fn cci<'py>(&self, py: Python<'py>, high: PyReadonlyArray1<'py, f64>, low: PyReadonlyArray1<'py, f64>, close: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Volume-synchronized Probability of Informed Trading (VPIN)
    ///
    /// VPIN is a real-time indicator of the probability of informed trading.
    /// It measures the imbalance between buy and sell volumes over a rolling window,
    /// helping to identify periods of informed trading activity.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `buy_volumes`: Array of buy volume data
    /// - `sell_volumes`: Array of sell volume data
    /// - `window`: Rolling window size for VPIN calculation
    ///
    /// # Returns
    /// `PyArray1<f64>` containing VPIN values (between 0 and 1)
    ///
    /// # Formula
    /// VPIN = |Buy Volume - Sell Volume| / (Buy Volume + Sell Volume)
    /// calculated over a rolling window
    ///
    /// # Performance Note
    /// This indicator benefits significantly from GPU acceleration for large datasets
    /// and is automatically optimized in the AdaptiveBackend.
    fn vpin<'py>(
        &self,
        py: Python<'py>,
        buy_volumes: PyReadonlyArray1<'py, f64>,
        sell_volumes: PyReadonlyArray1<'py, f64>,
        window: usize
    ) -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Ehlers SuperSmoother Filter
    ///
    /// SuperSmoother is a low-pass filter that provides superior smoothing with minimal lag.
    /// It uses a two-pole Butterworth filter design that eliminates aliasing and provides
    /// excellent noise reduction while preserving signal integrity.
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `data`: Array of input values (typically price data)
    /// - `period`: Filter period (must be >= 2)
    ///
    /// # Returns
    /// `PyArray1<f64>` containing SuperSmoother filtered values
    ///
    /// # Formula
    /// SuperSmoother uses coefficients calculated from:
    /// - a1 = exp(-1.414 * π / period)
    /// - b1 = 2 * a1 * cos(1.414 * π / period)
    /// - c1 = 1 - c2 - c3, c2 = b1, c3 = -a1²
    ///
    /// Filter equation: out\[i\] = c1 * (data\[i\] + data\[i-1\]) / 2 + c2 * out\[i-1\] + c3 * out\[i-2\]
    ///
    /// # Performance Note
    /// This indicator has sequential dependencies and does not benefit from GPU acceleration.
    /// All backends delegate to CPU implementation for optimal performance.
    fn supersmoother<'py>(&self, py: Python<'py>, data: PyReadonlyArray1<'py, f64>, period: usize)
        -> PyResult<Py<PyArray1<f64>>>;

    /// Calculate Ehlers Hilbert Transform
    ///
    /// The Hilbert Transform produces a complex-valued signal from a real-valued input,
    /// enabling the calculation of instantaneous amplitude and phase. This implementation
    /// follows Ehlers' FIR-based approach with a 4-step process:
    ///
    /// 1. **Roofing Filter**: 48-period high-pass + configurable low-pass SuperSmoother
    /// 2. **AGC Normalization**: Automatic Gain Control for the real component
    /// 3. **Quadrature Generation**: One-bar difference method
    /// 4. **SuperSmoother**: Applied to the imaginary component
    ///
    /// # Parameters
    /// - `py`: Python interpreter context
    /// - `data`: Array of input values (typically price data)
    /// - `lp_period`: Low-pass filter period for the roofing filter (commonly 10, 14, or 20)
    ///
    /// # Returns
    /// Tuple of (real_component, imaginary_component) as `PyArray1<f64>`
    ///
    /// # Algorithm Details
    /// - **High-pass period**: Fixed at 48 periods
    /// - **AGC decay factor**: Fixed at 0.991
    /// - **SuperSmoother period**: Fixed at 10 periods
    ///
    /// # Formula
    /// 1. Roofed = SuperSmoother(HighPass(data, 48), lp_period)
    /// 2. Real = AGC(Roofed, 0.991)
    /// 3. Quadrature = OneDifference(Real)
    /// 4. Imaginary = SuperSmoother(AGC(Quadrature, 0.991), 10)
    ///
    /// # Performance Note
    /// This indicator has mixed parallelization potential. The roofing filter stages
    /// can benefit from GPU acceleration, but AGC and SuperSmoother have sequential
    /// dependencies that are better suited for CPU computation.
    fn hilbert_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray1<'py, f64>, lp_period: usize)
        -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>;
}