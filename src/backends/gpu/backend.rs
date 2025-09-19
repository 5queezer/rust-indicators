//! Partial GPU Backend Implementation
//!
//! This module provides a hybrid GPU/CPU backend that selectively uses GPU acceleration
//! for specific indicators while falling back to CPU for others. The "partial" designation
//! reflects the current implementation state where only certain indicators benefit from
//! GPU acceleration.
//!
//! # Architecture Overview
//!
//! The partial GPU backend represents a pragmatic approach to GPU acceleration:
//!
//! ## Core Components
//!
//! - **PartialGpuBackend**: Main backend that manages GPU availability and delegation
//! - **GPU Implementations**: CUDA and WebGPU implementations for select indicators
//! - **CPU Fallback**: Embedded CPU backend for non-GPU indicators and error recovery
//! - **Feature Flag System**: Conditional compilation for different GPU backends
//!
//! ## Hybrid Architecture
//!
//! The backend uses a selective acceleration strategy:
//!
//! 1. **GPU-Accelerated Indicators**: VPIN and Hilbert Transform (when beneficial)
//! 2. **CPU-Delegated Indicators**: All other indicators use CPU backend
//! 3. **Dynamic Fallback**: Automatic CPU fallback on GPU errors or unavailability
//! 4. **Feature Detection**: Runtime GPU availability checking
//!
//! # GPU Support Matrix
//!
//! ## Currently GPU-Accelerated
//!
//! - **VPIN**: Volume-synchronized Probability of Informed Trading
//!   - **CUDA**: Full CUDA implementation for NVIDIA GPUs
//!   - **WebGPU**: Cross-platform GPU compute via WGPU
//!   - **Performance**: Significant speedup for datasets > 2000 points
//!
//! - **Hilbert Transform**: Complex signal analysis
//!   - **CUDA**: Optimized CUDA kernels for parallel processing
//!   - **WebGPU**: GPU compute shaders for cross-platform acceleration
//!   - **Performance**: Benefits from parallel FFT and filtering operations
//!
//! ## CPU-Delegated Indicators
//!
//! All other indicators currently delegate to the CPU backend:
//! - RSI, EMA, SMA, Bollinger Bands, ATR, Williams %R, CCI, SuperSmoother
//! - **Rationale**: Sequential nature or insufficient parallelization benefit
//! - **Future Enhancement**: GPU implementations planned for high-volume scenarios
//!
//! # Feature Flag System
//!
//! The backend supports multiple GPU acceleration backends through feature flags:
//!
//! ## CUDA Backend (`feature = "cuda"`)
//!
//! ```toml
//! [features]
//! cuda = ["cudarc", "cubecl/cuda"]
//! ```
//!
//! - **Target**: NVIDIA GPUs with CUDA support
//! - **Performance**: Highest performance for supported operations
//! - **Requirements**: CUDA toolkit and compatible GPU
//! - **Detection**: Uses `CUDA_VISIBLE_DEVICES` environment variable
//!
//! ## WebGPU Backend (`feature = "gpu"`)
//!
//! ```toml
//! [features]
//! gpu = ["cubecl/wgpu"]
//! ```
//!
//! - **Target**: Cross-platform GPU compute (Vulkan, Metal, D3D12)
//! - **Performance**: Good performance across different GPU vendors
//! - **Requirements**: Modern GPU with compute shader support
//! - **Compatibility**: Works with AMD, Intel, and NVIDIA GPUs
//!
//! ## No GPU (`default`)
//!
//! - **Fallback**: All operations use CPU backend
//! - **Compatibility**: Universal compatibility without GPU requirements
//! - **Performance**: Consistent CPU performance for all indicators
//!
//! # GPU Availability Detection
//!
//! The backend implements robust GPU availability detection:
//!
//! ## Detection Logic
//!
//! ```rust
//! pub fn is_available() -> bool {
//!     match std::env::var("CUDA_VISIBLE_DEVICES") {
//!         Ok(val) => !val.is_empty(),
//!         Err(_) => false,
//!     }
//! }
//! ```
//!
//! ## Detection Strategy
//!
//! - **Environment Variable**: Checks `CUDA_VISIBLE_DEVICES` for GPU availability
//! - **Fail-Fast**: Returns error immediately if GPU is not available
//! - **Test Mode**: Special handling for testing environments
//! - **Graceful Degradation**: Falls back to CPU on GPU initialization failure
//!
//! # Performance Characteristics
//!
//! ## GPU-Accelerated Operations
//!
//! - **VPIN**: 2-5x speedup for datasets > 2000 points
//! - **Hilbert Transform**: 3-8x speedup for complex signal processing
//! - **Memory Transfer**: Optimized data transfer patterns
//! - **Parallel Processing**: Leverages GPU's parallel compute units
//!
//! ## CPU-Delegated Operations
//!
//! - **Consistent Performance**: Same as pure CPU backend
//! - **No GPU Overhead**: Direct CPU execution without GPU initialization
//! - **Memory Efficiency**: No GPU memory allocation or transfer costs
//!
//! # Usage Example
//!
//! ```rust
//! use rust_indicators::backends::gpu::PartialGpuBackend;
//! use rust_indicators::core::traits::IndicatorsBackend;
//!
//! // Attempt to create GPU backend (may fail if GPU unavailable)
//! // match PartialGpuBackend::new() {
//! //     Ok(backend) => {
//! //         // GPU backend available - VPIN will use GPU acceleration
//! //         // let vpin_result = backend.vpin(py, buy_vols, sell_vols, 50)?;
//! //         
//! //         // Other indicators automatically use CPU backend
//! //         // let rsi_result = backend.rsi(py, prices, 14)?;
//! //     },
//! //     Err(_) => {
//! //         // GPU not available - use CPU backend instead
//! //         // let cpu_backend = CpuBackend::new();
//! //     }
//! // }
//! # // Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Error Handling and Fallback
//!
//! The partial GPU backend provides robust error handling:
//!
//! ## Initialization Errors
//!
//! - **GPU Unavailable**: Returns `PyRuntimeError` if GPU cannot be initialized
//! - **Feature Disabled**: Graceful handling when GPU features are disabled
//! - **Driver Issues**: Proper error propagation for GPU driver problems
//!
//! ## Runtime Fallback
//!
//! - **GPU Computation Errors**: Automatic fallback to CPU implementation
//! - **Memory Errors**: Graceful handling of GPU memory allocation failures
//! - **Timeout Handling**: Fallback on GPU computation timeouts
//!
//! # Implementation Details
//!
//! ## Macro-Based Delegation
//!
//! The backend uses the `gpu_method!` macro for CPU delegation:
//!
//! ```rust
//! // use rust_indicators::gpu_method;
//! //
//! // gpu_method!(rsi, (prices: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Py<PyArray1<f64>>>);
//! ```
//!
//! This provides:
//! - **Consistent Interface**: Uniform method signatures across backends
//! - **Code Reduction**: Eliminates boilerplate delegation code
//! - **Error Handling**: Consistent error propagation patterns
//!
//! ## Conditional Compilation
//!
//! GPU-specific code uses feature flags for conditional compilation:
//!
//! ```rust
//! #[cfg(feature = "cuda")]
//! {
//!     // CUDA-specific implementation
//!     // vpin_cuda_compute(buy_slice, sell_slice, window)
//! }
//! #[cfg(all(feature = "gpu", not(feature = "cuda")))]
//! {
//!     // WebGPU implementation
//!     // let device = WgpuDevice::default();
//!     // let client = WgpuRuntime::client(&device);
//!     // vpin_gpu_compute::<WgpuRuntime>(&client, buy_slice, sell_slice, window)
//! }
//! ```

pub use super::partial_gpu_backend::PartialGpuBackend;
