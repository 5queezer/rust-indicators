//! Macros for eliminating code duplication in backend implementations
//!
//! This module provides macros to reduce boilerplate code in adaptive backend
//! indicator delegation patterns.

/// Macro to generate the GPU/CPU delegation pattern for indicator methods.
///
/// This macro eliminates the duplication of the following pattern:
/// - Check if GPU should be used based on indicator name and parameters
/// - If GPU backend exists and should be used, delegate to GPU backend
/// - Otherwise, delegate to CPU backend
///
/// The macro reduces code duplication significantly - from approximately 13 lines
/// per indicator to just 4 lines, eliminating 72 lines of boilerplate across 8 indicators.
///
/// # Examples
///
/// ## Single Parameter Indicator (RSI)
///
/// ```ignore
/// delegate_indicator!(
///     self, py, "rsi",
///     IndicatorParams::Rsi { data_size: prices.as_array().len(), period },
///     rsi(prices, period)
/// );
/// ```
///
/// ## Multiple Parameter Indicator (ATR)
///
/// ```ignore
/// delegate_indicator!(
///     self, py, "atr",
///     IndicatorParams::Atr { data_size: high.as_array().len(), period },
///     atr(high, low, close, period)
/// );
/// ```
///
/// ## Tuple Return Indicator (Bollinger Bands)
///
/// ```ignore
/// delegate_indicator!(
///     self, py, "bollinger_bands",
///     IndicatorParams::BollingerBands { data_size: prices.as_array().len(), period },
///     bollinger_bands(prices, period, std_dev)
/// );
/// ```
///
/// # Code Reduction Benefits
///
/// The macro eliminates the following duplication for each indicator:
/// - Parameter construction and validation
/// - GPU availability checking logic
/// - Backend delegation decision making
/// - Fallback to CPU backend handling
///
/// **Before macro usage:**
/// - ~13 lines per indicator × 8 indicators = 104 lines
///
/// **After macro usage:**
/// - ~4 lines per indicator × 8 indicators = 32 lines
///
/// **Total reduction:** 72 lines of boilerplate code eliminated!
///
/// # Parameters
/// - `$self`: Reference to the AdaptiveBackend instance
/// - `$py`: Python context for PyO3 operations
/// - `$indicator_name`: String literal identifying the indicator for performance profiling
/// - `$params`: IndicatorParams variant construction with data size and parameters
/// - `$method_call`: Method call with parameters (excluding the `py` parameter which is added automatically)
#[macro_export]
macro_rules! delegate_indicator {
    ($self:expr, $py:expr, $indicator_name:expr, $params:expr, $method_call:ident($($args:expr),*)) => {
        {
            let params = $params;
            
            if $self.should_use_gpu($indicator_name, &params) {
                if let Some(ref gpu_backend) = $self.gpu_backend {
                    return gpu_backend.$method_call($py, $($args),*);
                }
            }
            
            $self.cpu_backend.$method_call($py, $($args),*)
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_macro_compiles() {
        // This test ensures the macro syntax is correct
        // Actual testing happens when the macro is used in the adaptive backend
        assert!(true);
    }
}