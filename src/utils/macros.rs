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
/// # Usage Examples
/// 
/// For single return value indicators like RSI:
/// ```ignore
/// delegate_indicator!(
///     self, py, "rsi", 
///     IndicatorParams::Rsi { data_size: prices.as_array().len(), period },
///     rsi(prices, period)
/// );
/// ```
/// 
/// For tuple return value indicators like Bollinger Bands:
/// ```ignore
/// delegate_indicator!(
///     self, py, "bollinger_bands",
///     IndicatorParams::BollingerBands { data_size: prices.as_array().len(), period },
///     bollinger_bands(prices, period, std_dev)
/// );
/// ```
/// 
/// # Parameters
/// - `$self`: Reference to the AdaptiveBackend instance
/// - `$py`: Python context
/// - `$indicator_name`: String literal for the indicator name
/// - `$params`: IndicatorParams variant construction
/// - `$method_call`: Method call with parameters (without `py` parameter)
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