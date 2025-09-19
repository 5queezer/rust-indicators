//! # Memory Optimization Utilities
//!
//! Memory optimization and allocation reduction utilities for ML components
//! to improve performance and reduce memory footprint across all models.

use crate::ml::components::cross_validation::CVSplitsOutput;
use numpy::ndarray;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Memory pool for reusing frequently allocated objects
pub struct MemoryPool<T> {
    pool: Arc<Mutex<Vec<T>>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool with a factory function
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
            factory: Box::new(factory),
        }
    }

    /// Get an object from the pool or create a new one
    pub fn get(&self) -> PooledObject<T> {
        let mut pool = self.pool.lock().unwrap();
        let object = pool.pop().unwrap_or_else(|| (self.factory)());
        PooledObject {
            object: Some(object),
            pool: Arc::clone(&self.pool),
        }
    }

    /// Get the current pool size
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T> {
    object: Option<T>,
    pool: Arc<Mutex<Vec<T>>>,
}

impl<T> PooledObject<T> {
    /// Get a reference to the pooled object
    pub fn get(&self) -> &T {
        self.object.as_ref().unwrap()
    }

    /// Get a mutable reference to the pooled object
    pub fn get_mut(&mut self) -> &mut T {
        self.object.as_mut().unwrap()
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            if let Ok(mut pool) = self.pool.lock() {
                pool.push(object);
            }
        }
    }
}

/// Shared computation cache for expensive operations
pub struct ComputationCache<K, V> {
    cache: Arc<Mutex<HashMap<K, V>>>,
    max_size: usize,
}

impl<K, V> ComputationCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new computation cache with maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
        }
    }

    /// Get a cached value or compute it
    pub fn get_or_compute<F>(&self, key: K, compute_fn: F) -> V
    where
        F: FnOnce() -> V,
    {
        let mut cache = self.cache.lock().unwrap();

        if let Some(value) = cache.get(&key) {
            return value.clone();
        }

        // Compute the value
        let value = compute_fn();

        // Check if we need to evict entries
        if cache.len() >= self.max_size {
            // Simple LRU: remove first entry (not optimal but simple)
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, value.clone());
        value
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

/// Memory-efficient array operations
pub struct ArrayOperations;

impl ArrayOperations {
    /// In-place array normalization to avoid allocations
    pub fn normalize_inplace(arr: &mut ndarray::Array1<f32>) {
        let sum = arr.sum();
        if sum != 0.0 {
            arr.mapv_inplace(|x| x / sum);
        }
    }

    /// In-place array standardization (zero mean, unit variance)
    pub fn standardize_inplace(arr: &mut ndarray::Array1<f32>) {
        let mean = arr.mean().unwrap_or(0.0);
        let std = arr.std(0.0);

        if std != 0.0 {
            arr.mapv_inplace(|x| (x - mean) / std);
        }
    }

    /// Reuse array for multiple operations
    pub fn reuse_array_for_computation<F, R>(
        mut arr: ndarray::Array1<f32>,
        computation: F,
    ) -> (ndarray::Array1<f32>, R)
    where
        F: FnOnce(&mut ndarray::Array1<f32>) -> R,
    {
        let result = computation(&mut arr);
        (arr, result)
    }

    /// Memory-efficient dot product without intermediate allocations
    pub fn efficient_dot_product(
        a: &ndarray::ArrayView1<f32>,
        b: &ndarray::ArrayView1<f32>,
    ) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// In-place element-wise operations
    pub fn apply_inplace<F>(arr: &mut ndarray::Array1<f32>, f: F)
    where
        F: Fn(f32) -> f32,
    {
        arr.mapv_inplace(f);
    }
}

/// Memory-efficient feature processing
pub struct FeatureProcessor {
    temp_buffer: Vec<f32>,
    computation_cache: ComputationCache<String, Vec<f32>>,
}

impl FeatureProcessor {
    /// Create a new feature processor with buffer size
    pub fn new(buffer_size: usize, cache_size: usize) -> Self {
        Self {
            temp_buffer: Vec::with_capacity(buffer_size),
            computation_cache: ComputationCache::new(cache_size),
        }
    }

    /// Process features in-place to avoid allocations
    pub fn process_features_inplace(
        &mut self,
        features: &mut ndarray::Array2<f32>,
        _processor_id: &str,
    ) -> Result<(), String> {
        let (n_samples, n_features) = features.dim();

        // Reuse buffer for temporary computations
        self.temp_buffer.clear();
        self.temp_buffer.reserve(n_features);

        for i in 0..n_samples {
            let mut row = features.row_mut(i);

            // Example processing: normalize each row
            let sum: f32 = row.iter().sum();
            if sum != 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(())
    }

    /// Cache expensive feature computations
    pub fn get_cached_features<F>(&self, key: String, compute_fn: F) -> Vec<f32>
    where
        F: FnOnce() -> Vec<f32>,
    {
        self.computation_cache.get_or_compute(key, compute_fn)
    }

    /// Clear all caches and buffers
    pub fn clear(&mut self) {
        self.temp_buffer.clear();
        self.computation_cache.clear();
    }
}

/// Memory-efficient cross-validation utilities
pub struct CVMemoryManager {
    split_cache: ComputationCache<(usize, usize, u32), CVSplitsOutput>,
    score_buffer: Vec<f32>,
}

impl CVMemoryManager {
    pub fn new(cache_size: usize) -> Self {
        Self {
            split_cache: ComputationCache::new(cache_size),
            score_buffer: Vec::new(),
        }
    }

    /// Get cached CV splits or compute them
    pub fn get_cv_splits<F>(
        &self,
        n_samples: usize,
        n_splits: usize,
        embargo_pct: f32,
        compute_fn: F,
    ) -> Vec<(Vec<usize>, Vec<usize>)>
    where
        F: FnOnce() -> Vec<(Vec<usize>, Vec<usize>)>,
    {
        // Convert f32 to u32 for hashing (multiply by 10000 to preserve precision)
        let embargo_key = (embargo_pct * 10000.0) as u32;
        let key = (n_samples, n_splits, embargo_key);
        self.split_cache.get_or_compute(key, compute_fn)
    }

    /// Reuse score buffer for CV computations
    pub fn with_score_buffer<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<f32>) -> R,
    {
        self.score_buffer.clear();
        f(&mut self.score_buffer)
    }
}

/// Global memory optimization settings
pub struct MemoryOptimizer {
    feature_pool: MemoryPool<Vec<f32>>,
    array_pool: MemoryPool<ndarray::Array1<f32>>,
    cv_manager: CVMemoryManager,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer with default settings
    pub fn new() -> Self {
        Self {
            feature_pool: MemoryPool::new(|| Vec::with_capacity(1000)),
            array_pool: MemoryPool::new(|| ndarray::Array1::zeros(100)),
            cv_manager: CVMemoryManager::new(50),
        }
    }

    /// Get a pooled feature vector
    pub fn get_feature_vector(&self) -> PooledObject<Vec<f32>> {
        self.feature_pool.get()
    }

    /// Get a pooled array
    pub fn get_array(&self) -> PooledObject<ndarray::Array1<f32>> {
        self.array_pool.get()
    }

    /// Get CV memory manager
    pub fn cv_manager(&mut self) -> &mut CVMemoryManager {
        &mut self.cv_manager
    }

    /// Clear all pools and caches
    pub fn clear_all(&mut self) {
        self.feature_pool.clear();
        self.array_pool.clear();
        self.cv_manager.split_cache.clear();
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            feature_pool_size: self.feature_pool.size(),
            array_pool_size: self.array_pool.size(),
            cv_cache_size: self.cv_manager.split_cache.size(),
        }
    }
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub feature_pool_size: usize,
    pub array_pool_size: usize,
    pub cv_cache_size: usize,
}

impl MemoryStats {
    pub fn total_cached_objects(&self) -> usize {
        self.feature_pool_size + self.array_pool_size + self.cv_cache_size
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_MEMORY_OPTIMIZER: Arc<Mutex<MemoryOptimizer>> =
        Arc::new(Mutex::new(MemoryOptimizer::new()));
}

/// Get the global memory optimizer
pub fn global_memory_optimizer() -> Arc<Mutex<MemoryOptimizer>> {
    Arc::clone(&GLOBAL_MEMORY_OPTIMIZER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(|| Vec::<i32>::new());

        {
            let mut obj1 = pool.get();
            obj1.get_mut().push(42);
            assert_eq!(obj1.get()[0], 42);
        } // obj1 is returned to pool here

        assert_eq!(pool.size(), 1);

        let obj2 = pool.get();
        assert_eq!(obj2.get().len(), 1); // Reused the previous vector
    }

    #[test]
    fn test_computation_cache() {
        let cache = ComputationCache::new(2);

        let result1 = cache.get_or_compute("key1".to_string(), || 42);
        assert_eq!(result1, 42);

        let result2 = cache.get_or_compute("key1".to_string(), || 100);
        assert_eq!(result2, 42); // Should return cached value

        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_array_operations() {
        let mut arr = ndarray::Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        ArrayOperations::normalize_inplace(&mut arr);

        let sum: f32 = arr.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_feature_processor() {
        let mut processor = FeatureProcessor::new(100, 10);
        let mut features =
            ndarray::Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        processor
            .process_features_inplace(&mut features, "test")
            .unwrap();

        // Check that rows are normalized
        let row1_sum: f32 = features.row(0).sum();
        let row2_sum: f32 = features.row(1).sum();
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_optimizer() {
        let optimizer = MemoryOptimizer::new();

        {
            let _vec = optimizer.get_feature_vector();
            let _arr = optimizer.get_array();
        }

        let stats = optimizer.get_stats();
        assert_eq!(stats.feature_pool_size, 1);
        assert_eq!(stats.array_pool_size, 1);
    }
}
