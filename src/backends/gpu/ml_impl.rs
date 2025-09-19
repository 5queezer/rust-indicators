use crate::backends::cpu::ml_backend::CpuMLBackend;
use crate::backends::gpu::partial_gpu_backend::PartialGpuBackend;
use crate::ml::traits::{BatchPredictionOutput, MLBackend};
use anyhow::Result;
use log::{error, info, warn};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
use crate::backends::gpu::implementations::{
    accumulate_gradient_kernel, batch_prediction_kernel, calculate_pattern_ensemble_weights_kernel,
    calculate_prediction_error_kernel, evaluate_pattern_kernel, gradient_descent_update_kernel,
};
#[cfg(feature = "cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime};
#[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
use cubecl::prelude::{ArrayArg, ComputeClient, CubeCount, CubeDim, ScalarArg};
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

impl MLBackend for PartialGpuBackend {
    fn train_model<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
        pattern_features: Option<PyReadonlyArray2<'py, f32>>,
        price_features: Option<PyReadonlyArray2<'py, f32>>,
        pattern_names: Option<Vec<String>>,
    ) -> PyResult<HashMap<String, f32>> {
        self.ml_backend.train_model(
            py,
            features,
            labels,
            pattern_features,
            price_features,
            pattern_names,
        )
    }

    fn train_trading_fold<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i32>,
        train_idx: &[usize],
        test_idx: &[usize],
        learning_rate: f32,
        sample_weights: PyReadonlyArray1<'py, f32>,
        n_features: usize,
    ) -> PyResult<(f32, Vec<f32>)> {
        let operation_name = "train_trading_fold";
        if PartialGpuBackend::is_available() {
            self.performance_monitor
                .track_operation(operation_name, || {
                    match self._gpu_train_trading_fold(
                        py,
                        x,
                        y,
                        train_idx,
                        test_idx,
                        learning_rate,
                        sample_weights,
                        n_features,
                    ) {
                        Ok(result) => {
                            self.gpu_failure_count
                                .store(0, std::sync::atomic::Ordering::SeqCst);
                            Ok(result)
                        }
                        Err(e) => {
                            error!("GPU train_trading_fold failed: {}. Falling back to CPU.", e);
                            self.gpu_failure_count
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.performance_monitor.record_fallback(operation_name);
                            self.ml_backend.train_trading_fold(
                                py,
                                x,
                                y,
                                train_idx,
                                test_idx,
                                learning_rate,
                                sample_weights,
                                n_features,
                            )
                        }
                    }
                })
        } else {
            warn!("GPU not available for train_trading_fold, falling back to CPU.");
            self.ml_backend.train_trading_fold(
                py,
                x,
                y,
                train_idx,
                test_idx,
                learning_rate,
                sample_weights,
                n_features,
            )
        }
    }

    fn train_final_trading_model<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i32>,
        learning_rate: f32,
        sample_weights: PyReadonlyArray1<'py, f32>,
        n_features: usize,
    ) -> PyResult<Vec<f32>> {
        let operation_name = "train_final_trading_model";
        if PartialGpuBackend::is_available() {
            self.performance_monitor
                .track_operation(operation_name, || {
                    match self._gpu_train_final_trading_model(
                        py,
                        x,
                        y,
                        learning_rate,
                        sample_weights,
                        n_features,
                    ) {
                        Ok(weights) => {
                            self.gpu_failure_count
                                .store(0, std::sync::atomic::Ordering::SeqCst);
                            self.model_weights = weights.clone();
                            Ok(weights)
                        }
                        Err(e) => {
                            error!(
                                "GPU train_final_trading_model failed: {}. Falling back to CPU.",
                                e
                            );
                            self.gpu_failure_count
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.performance_monitor.record_fallback(operation_name);
                            self.ml_backend.train_final_trading_model(
                                py,
                                x,
                                y,
                                learning_rate,
                                sample_weights,
                                n_features,
                            )
                        }
                    }
                })
        } else {
            warn!("GPU not available for train_final_trading_model, falling back to CPU.");
            self.ml_backend.train_final_trading_model(
                py,
                x,
                y,
                learning_rate,
                sample_weights,
                n_features,
            )
        }
    }

    fn get_model_weights<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        if self.model_weights.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Model not trained, no weights available.",
            ));
        }
        Ok(PyArray1::from_vec(py, self.model_weights.clone())
            .to_owned()
            .into())
    }

    fn predict_with_confidence<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(i32, f32)> {
        self.ml_backend.predict_with_confidence(py, features)
    }

    fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f32>>> {
        self.ml_backend.get_feature_importance(py)
    }

    fn is_trained(&self) -> bool {
        self.ml_backend.is_trained()
    }

    fn reset_model(&mut self) {
        self.ml_backend.reset_model();
        self.model_weights.clear();
    }

    fn evaluate_pattern_fold<'py>(
        &self,
        py: Python<'py>,
        pattern_features: PyReadonlyArray2<'py, f32>,
        price_features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
        test_idx: &[usize],
        pattern_names: &[String],
        pattern_weights: &HashMap<String, f32>,
        confidence_threshold: f32,
    ) -> PyResult<f32> {
        let operation_name = "evaluate_pattern_fold";
        if PartialGpuBackend::is_available() {
            self.performance_monitor
                .track_operation(operation_name, || {
                    match self._gpu_evaluate_pattern_fold(
                        pattern_features,
                        labels,
                        test_idx,
                        pattern_names,
                        pattern_weights,
                    ) {
                        Ok(accuracy) => {
                            self.gpu_failure_count
                                .store(0, std::sync::atomic::Ordering::SeqCst);
                            Ok(accuracy)
                        }
                        Err(e) => {
                            error!(
                                "GPU evaluate_pattern_fold failed: {}. Falling back to CPU.",
                                e
                            );
                            self.gpu_failure_count
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.performance_monitor.record_fallback(operation_name);
                            self.ml_backend.evaluate_pattern_fold(
                                py,
                                pattern_features,
                                price_features,
                                labels,
                                test_idx,
                                pattern_names,
                                pattern_weights,
                                confidence_threshold,
                            )
                        }
                    }
                })
        } else {
            warn!("GPU not available for evaluate_pattern_fold, falling back to CPU.");
            self.ml_backend.evaluate_pattern_fold(
                py,
                pattern_features,
                price_features,
                labels,
                test_idx,
                pattern_names,
                pattern_weights,
                confidence_threshold,
            )
        }
    }

    fn calculate_pattern_ensemble_weights<'py>(
        &self,
        py: Python<'py>,
        pattern_importance: &HashMap<String, f32>,
        pattern_names: &[String],
    ) -> PyResult<HashMap<String, f32>> {
        let operation_name = "calculate_pattern_ensemble_weights";
        if PartialGpuBackend::is_available() {
            self.performance_monitor.track_operation(operation_name, || {
                match self._gpu_calculate_pattern_ensemble_weights(pattern_importance, pattern_names) {
                    Ok(weights) => {
                        self.gpu_failure_count.store(0, std::sync::atomic::Ordering::SeqCst);
                        Ok(weights)
                    }
                    Err(e) => {
                        error!("GPU calculate_pattern_ensemble_weights failed: {}. Falling back to CPU.", e);
                        self.gpu_failure_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        self.performance_monitor.record_fallback(operation_name);
                        self.ml_backend.calculate_pattern_ensemble_weights(py, pattern_importance, pattern_names)
                    }
                }
            })
        } else {
            warn!("GPU not available for calculate_pattern_ensemble_weights, falling back to CPU.");
            self.ml_backend.calculate_pattern_ensemble_weights(
                py,
                pattern_importance,
                pattern_names,
            )
        }
    }

    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<BatchPredictionOutput> {
        let operation_name = "predict_batch";
        if PartialGpuBackend::is_available() {
            self.performance_monitor
                .track_operation(operation_name, || {
                    match self._gpu_predict_batch(py, features) {
                        Ok(output) => {
                            self.gpu_failure_count
                                .store(0, std::sync::atomic::Ordering::SeqCst);
                            Ok(output)
                        }
                        Err(e) => {
                            error!("GPU predict_batch failed: {}. Falling back to CPU.", e);
                            self.gpu_failure_count
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.performance_monitor.record_fallback(operation_name);
                            self.ml_backend.predict_batch(py, features)
                        }
                    }
                })
        } else {
            warn!("GPU not available for predict_batch, falling back to CPU.");
            self.ml_backend.predict_batch(py, features)
        }
    }

    fn predict_probabilities<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.ml_backend.predict_probabilities(py, features)
    }

    fn get_prediction_explanation<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        self.ml_backend.get_prediction_explanation(py, features)
    }

    fn set_confidence_threshold_unchecked(&mut self, threshold: f32) {
        self.ml_backend
            .set_confidence_threshold_unchecked(threshold);
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.ml_backend.get_confidence_threshold()
    }
}

impl PartialGpuBackend {
    #[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
    fn _gpu_train_trading_fold<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i32>,
        _train_idx: &[usize],
        test_idx: &[usize],
        learning_rate: f32,
        sample_weights: PyReadonlyArray1<'py, f32>,
        n_features: usize,
    ) -> Result<(f32, Vec<f32>)> {
        info!("Attempting GPU-accelerated train_trading_fold.");
        let x_array = x.as_array();
        let y_array = y.as_array();

        let x_f32: Vec<f32> = x_array.iter().map(|&val| val).collect();
        let y_i32: Vec<i32> = y_array.iter().map(|&val| val).collect();

        let mut ml_backend_clone = self.ml_backend.clone();
        let weights = ml_backend_clone.train_final_trading_model(
            py,
            x,
            y,
            learning_rate,
            sample_weights,
            n_features,
        )?;
        let feature_importance = weights.clone();

        let mut correct = 0;
        let mut total = 0;

        for &idx in test_idx {
            let features_slice = &x_f32[idx * n_features..(idx + 1) * n_features];
            let weighted_sum = features_slice
                .iter()
                .zip(&weights)
                .map(|(&f, &w)| f * w)
                .sum::<f32>();
            let normalized = weighted_sum.tanh();

            let pred_class = if normalized > 0.15 {
                2
            } else if normalized < -0.15 {
                0
            } else {
                1
            };

            if pred_class == y_i32[idx] {
                correct += 1;
            }
            total += 1;
        }

        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        Ok((accuracy, feature_importance))
    }

    #[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
    fn _gpu_train_final_trading_model<'py>(
        &self,
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, i32>,
        learning_rate: f32,
        sample_weights: PyReadonlyArray1<'py, f32>,
        n_features: usize,
    ) -> Result<Vec<f32>> {
        info!("Attempting GPU-accelerated train_final_trading_model.");
        let x_array = x.as_array();
        let y_array = y.as_array();
        let sample_weights_array = sample_weights.as_array();

        let (n_samples, _) = x_array.dim();

        let x_f32: Vec<f32> = x_array.iter().map(|&val| val).collect();
        let y_i32: Vec<i32> = y_array.iter().map(|&val| val).collect();
        let sample_weights_f32: Vec<f32> = sample_weights_array.iter().map(|&val| val).collect();

        let client = {
            #[cfg(feature = "cuda")]
            {
                let device = CudaDevice::new(0);
                CudaRuntime::client(&device)
            }
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            {
                let device = WgpuDevice::default();
                WgpuRuntime::client(&device)
            }
        };

        let x_gpu = self
            .memory_pool
            .alloc(n_samples * n_features * core::mem::size_of::<f32>());
        client.write(x_gpu.binding(), bytemuck::cast_slice(&x_f32));
        let y_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<i32>());
        client.write(y_gpu.binding(), bytemuck::cast_slice(&y_i32));
        let sample_weights_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<f32>());
        client.write(
            sample_weights_gpu.binding(),
            bytemuck::cast_slice(&sample_weights_f32),
        );
        let mut weights_gpu = self
            .memory_pool
            .alloc(n_features * core::mem::size_of::<f32>());
        client.write(
            weights_gpu.binding(),
            bytemuck::cast_slice(&vec![0.01f32; n_features]),
        );
        let mut gradient_gpu = self
            .memory_pool
            .alloc(n_features * core::mem::size_of::<f32>());
        let mut errors_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<f32>());
        let mut predictions_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<f32>());

        let threads_per_block = 256u32;
        let blocks_samples = (n_samples as u32 + threads_per_block - 1) / threads_per_block;
        let blocks_features = (n_features as u32 + threads_per_block - 1) / threads_per_block;

        let epochs = 100;

        for _ in 0..epochs {
            unsafe {
                macro_rules! launch_kernel {
                    ($runtime:ty) => {
                        calculate_prediction_error_kernel::launch::<$runtime>(
                            &client,
                            CubeCount::Static(blocks_samples, 1, 1),
                            CubeDim::new(threads_per_block, 1, 1),
                            ArrayArg::new(&x_gpu, n_samples * n_features),
                            ArrayArg::new(&weights_gpu, n_features),
                            ArrayArg::new(&y_gpu, n_samples),
                            ArrayArg::new(&mut predictions_gpu, n_samples),
                            ArrayArg::new(&mut errors_gpu, n_samples),
                            ScalarArg::new(n_features as u32),
                            ScalarArg::new(n_samples as u32),
                        );
                    };
                }
                #[cfg(feature = "cuda")]
                launch_kernel!(CudaRuntime);
                #[cfg(all(feature = "gpu", not(feature = "cuda")))]
                launch_kernel!(WgpuRuntime);

                macro_rules! launch_kernel {
                    ($runtime:ty) => {
                        accumulate_gradient_kernel::launch::<$runtime>(
                            &client,
                            CubeCount::Static(blocks_features, 1, 1),
                            CubeDim::new(threads_per_block, 1, 1),
                            ArrayArg::new(&x_gpu, n_samples * n_features),
                            ArrayArg::new(&errors_gpu, n_samples),
                            ArrayArg::new(&sample_weights_gpu, n_samples),
                            ArrayArg::new(&mut gradient_gpu, n_features),
                            ScalarArg::new(n_features as u32),
                            ScalarArg::new(n_samples as u32),
                        );
                    };
                }
                #[cfg(feature = "cuda")]
                launch_kernel!(CudaRuntime);
                #[cfg(all(feature = "gpu", not(feature = "cuda")))]
                launch_kernel!(WgpuRuntime);

                macro_rules! launch_kernel {
                    ($runtime:ty) => {
                        gradient_descent_update_kernel::launch::<$runtime>(
                            &client,
                            CubeCount::Static(blocks_features, 1, 1),
                            CubeDim::new(threads_per_block, 1, 1),
                            CubeDim::new(threads_per_block, 1, 1),
                            ArrayArg::new(&mut weights_gpu, n_features),
                            ArrayArg::new(&gradient_gpu, n_features),
                            ScalarArg::new(learning_rate / n_samples as f32),
                            ScalarArg::new(n_features as u32),
                        );
                    };
                }
                #[cfg(feature = "cuda")]
                launch_kernel!(CudaRuntime);
                #[cfg(all(feature = "gpu", not(feature = "cuda")))]
                launch_kernel!(WgpuRuntime);
            }
        }

        let weights_bytes = client.read(weights_gpu.binding());
        let result = bytemuck::cast_slice(&weights_bytes).to_vec();

        self.memory_pool.dealloc(x_gpu);
        self.memory_pool.dealloc(y_gpu);
        self.memory_pool.dealloc(sample_weights_gpu);
        self.memory_pool.dealloc(weights_gpu);
        self.memory_pool.dealloc(gradient_gpu);
        self.memory_pool.dealloc(errors_gpu);
        self.memory_pool.dealloc(predictions_gpu);

        Ok(result)
    }

    #[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
    fn _gpu_evaluate_pattern_fold(
        &self,
        pattern_features: PyReadonlyArray2<f32>,
        labels: PyReadonlyArray1<i32>,
        test_idx: &[usize],
        pattern_names: &[String],
        pattern_weights: &HashMap<String, f32>,
    ) -> Result<f32> {
        info!("Attempting GPU-accelerated evaluate_pattern_fold.");
        let pattern_features_array = pattern_features.as_array();
        let labels_array = labels.as_array();

        let (n_samples, n_patterns) = pattern_features_array.dim();

        let pattern_features_f32: Vec<f32> =
            pattern_features_array.iter().map(|&val| val).collect();
        let labels_i32: Vec<i32> = labels_array.iter().map(|&val| val).collect();

        let mut pattern_weights_vec = vec![0.0f32; n_patterns];
        for (i, pattern_name) in pattern_names.iter().enumerate() {
            if let Some(&weight) = pattern_weights.get(pattern_name) {
                pattern_weights_vec[i] = weight;
            }
        }

        let client = {
            #[cfg(feature = "cuda")]
            {
                let device = CudaDevice::new(0);
                CudaRuntime::client(&device)
            }
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            {
                let device = WgpuDevice::default();
                WgpuRuntime::client(&device)
            }
        };

        let pattern_features_gpu = self
            .memory_pool
            .alloc(n_samples * n_patterns * core::mem::size_of::<f32>());
        client.write(
            pattern_features_gpu.binding(),
            bytemuck::cast_slice(&pattern_features_f32),
        );
        let pattern_weights_gpu = self
            .memory_pool
            .alloc(n_patterns * core::mem::size_of::<f32>());
        client.write(
            pattern_weights_gpu.binding(),
            bytemuck::cast_slice(&pattern_weights_vec),
        );
        let mut output_scores_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<f32>());

        let threads_per_block = 256u32;
        let blocks_samples = (n_samples as u32 + threads_per_block - 1) / threads_per_block;

        unsafe {
            macro_rules! launch_kernel {
                ($runtime:ty) => {
                    evaluate_pattern_kernel::launch::<$runtime>(
                        &client,
                        CubeCount::Static(blocks_samples, 1, 1),
                        CubeDim::new(threads_per_block, 1, 1),
                        ArrayArg::new(&pattern_features_gpu, n_samples * n_patterns),
                        ArrayArg::new(&pattern_weights_gpu, n_patterns),
                        ArrayArg::new(&mut output_scores_gpu, n_samples),
                        ScalarArg::new(n_patterns as u32),
                        ScalarArg::new(n_samples as u32),
                    );
                };
            }
            #[cfg(feature = "cuda")]
            launch_kernel!(CudaRuntime);
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            launch_kernel!(WgpuRuntime);
        }

        let output_scores_bytes = client.read(output_scores_gpu.binding());
        let output_scores_f32: &[f32] = bytemuck::cast_slice(&output_scores_bytes);

        self.memory_pool.dealloc(pattern_features_gpu);
        self.memory_pool.dealloc(pattern_weights_gpu);
        self.memory_pool.dealloc(output_scores_gpu);

        let mut correct = 0;
        let mut total = 0;
        for &idx in test_idx {
            let pred_score = output_scores_f32[idx];
            let pred_class = if pred_score > 0.15 {
                2
            } else if pred_score < -0.15 {
                0
            } else {
                1
            };

            if pred_class == labels_i32[idx] {
                correct += 1;
            }
            total += 1;
        }

        Ok(if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        })
    }

    #[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
    fn _gpu_calculate_pattern_ensemble_weights(
        &self,
        pattern_importance: &HashMap<String, f32>,
        pattern_names: &[String],
    ) -> Result<HashMap<String, f32>> {
        info!("Attempting GPU-accelerated calculate_pattern_ensemble_weights.");
        let n_patterns = pattern_names.len();
        let mut importance_vec = vec![0.0f32; n_patterns];
        for (i, name) in pattern_names.iter().enumerate() {
            if let Some(&importance) = pattern_importance.get(name) {
                importance_vec[i] = importance;
            }
        }

        let client = {
            #[cfg(feature = "cuda")]
            {
                let device = CudaDevice::new(0);
                CudaRuntime::client(&device)
            }
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            {
                let device = WgpuDevice::default();
                WgpuRuntime::client(&device)
            }
        };

        let importance_gpu = self
            .memory_pool
            .alloc(n_patterns * core::mem::size_of::<f32>());
        client.write(
            importance_gpu.binding(),
            bytemuck::cast_slice(&importance_vec),
        );
        let mut output_weights_gpu = self
            .memory_pool
            .alloc(n_patterns * core::mem::size_of::<f32>());

        let threads_per_block = 256u32;
        let blocks = (n_patterns as u32 + threads_per_block - 1) / threads_per_block;

        unsafe {
            macro_rules! launch_kernel {
                ($runtime:ty) => {
                    calculate_pattern_ensemble_weights_kernel::launch::<$runtime>(
                        &client,
                        CubeCount::Static(blocks, 1, 1),
                        CubeDim::new(threads_per_block, 1, 1),
                        ArrayArg::new(&importance_gpu, n_patterns),
                        ArrayArg::new(&mut output_weights_gpu, n_patterns),
                        ScalarArg::new(n_patterns as u32),
                    );
                };
            }
            #[cfg(feature = "cuda")]
            launch_kernel!(CudaRuntime);
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            launch_kernel!(WgpuRuntime);
        }

        let output_weights_bytes = client.read(output_weights_gpu.binding());
        let output_weights_f32: &[f32] = bytemuck::cast_slice(&output_weights_bytes);

        let mut weights_map = HashMap::new();
        for (i, name) in pattern_names.iter().enumerate() {
            weights_map.insert(name.clone(), output_weights_f32[i]);
        }

        self.memory_pool.dealloc(importance_gpu);
        self.memory_pool.dealloc(output_weights_gpu);

        Ok(weights_map)
    }

    #[cfg(any(feature = "cuda", all(feature = "gpu", not(feature = "cuda"))))]
    fn _gpu_predict_batch<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> Result<BatchPredictionOutput> {
        info!("Attempting GPU-accelerated predict_batch.");
        let features_array = features.as_array();
        let (n_samples, n_features) = features_array.dim();

        let features_f32: Vec<f32> = features_array.iter().map(|&val| val).collect();

        let weights_py = self.get_model_weights(py)?;
        let weights_vec: Vec<f32> = weights_py.extract(py)?;

        let client = {
            #[cfg(feature = "cuda")]
            {
                let device = CudaDevice::new(0);
                CudaRuntime::client(&device)
            }
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            {
                let device = WgpuDevice::default();
                WgpuRuntime::client(&device)
            }
        };

        let features_gpu = self
            .memory_pool
            .alloc(n_samples * n_features * core::mem::size_of::<f32>());
        client.write(features_gpu.binding(), bytemuck::cast_slice(&features_f32));
        let weights_gpu = self
            .memory_pool
            .alloc(n_features * core::mem::size_of::<f32>());
        client.write(weights_gpu.binding(), bytemuck::cast_slice(&weights_vec));
        let mut predictions_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<i32>());
        let mut confidences_gpu = self
            .memory_pool
            .alloc(n_samples * core::mem::size_of::<f32>());

        let threads_per_block = 256u32;
        let blocks_samples = (n_samples as u32 + threads_per_block - 1) / threads_per_block;

        unsafe {
            macro_rules! launch_kernel {
                ($runtime:ty) => {
                    batch_prediction_kernel::launch::<$runtime>(
                        &client,
                        CubeCount::Static(blocks_samples, 1, 1),
                        CubeDim::new(threads_per_block, 1, 1),
                        ArrayArg::new(&features_gpu, n_samples * n_features),
                        ArrayArg::new(&weights_gpu, n_features),
                        ArrayArg::new(&mut predictions_gpu, n_samples),
                        ArrayArg::new(&mut confidences_gpu, n_samples),
                        ScalarArg::new(n_features as u32),
                        ScalarArg::new(n_samples as u32),
                    );
                };
            }
            #[cfg(feature = "cuda")]
            launch_kernel!(CudaRuntime);
            #[cfg(all(feature = "gpu", not(feature = "cuda")))]
            launch_kernel!(WgpuRuntime);
        }

        let predictions_bytes = client.read(predictions_gpu.binding());
        let predictions_i32: &[i32] = bytemuck::cast_slice(&predictions_bytes);
        let confidences_bytes = client.read(confidences_gpu.binding());
        let confidences_f32: &[f32] = bytemuck::cast_slice(&confidences_bytes);

        let result = (
            PyArray1::from_slice(py, predictions_i32).to_owned().into(),
            PyArray1::from_slice(py, confidences_f32).to_owned().into(),
        );

        self.memory_pool.dealloc(features_gpu);
        self.memory_pool.dealloc(weights_gpu);
        self.memory_pool.dealloc(predictions_gpu);
        self.memory_pool.dealloc(confidences_gpu);

        Ok(result)
    }
}
