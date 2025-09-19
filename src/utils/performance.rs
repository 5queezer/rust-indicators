//! Performance Monitoring Utilities
//!
//! This module provides tools for tracking and reporting performance metrics
//! for different backend operations, particularly for comparing CPU vs. GPU execution.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A struct to hold performance metrics for a single operation.
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    pub execution_times: Vec<Duration>,
    pub success_count: u64,
    pub failure_count: u64,
    pub fallback_count: u64,
}

impl OperationMetrics {
    pub fn new() -> Self {
        Self {
            execution_times: Vec::new(),
            success_count: 0,
            failure_count: 0,
            fallback_count: 0,
        }
    }

    pub fn record_execution(&mut self, time: Duration) {
        self.execution_times.push(time);
        self.success_count += 1;
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }

    pub fn record_fallback(&mut self) {
        self.fallback_count += 1;
    }

    pub fn average_execution_time(&self) -> Duration {
        if self.execution_times.is_empty() {
            return Duration::from_secs(0);
        }
        let total: Duration = self.execution_times.iter().sum();
        total / self.execution_times.len() as u32
    }
}

/// A thread-safe performance monitor to track metrics for multiple operations.
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<HashMap<String, OperationMetrics>>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn track_operation<F, R>(&self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        let mut metrics = self.metrics.lock().unwrap();
        let op_metrics = metrics
            .entry(name.to_string())
            .or_insert_with(OperationMetrics::new);
        op_metrics.record_execution(duration);

        result
    }

    pub fn record_failure(&self, name: &str) {
        let mut metrics = self.metrics.lock().unwrap();
        let op_metrics = metrics
            .entry(name.to_string())
            .or_insert_with(OperationMetrics::new);
        op_metrics.record_failure();
    }

    pub fn record_fallback(&self, name: &str) {
        let mut metrics = self.metrics.lock().unwrap();
        let op_metrics = metrics
            .entry(name.to_string())
            .or_insert_with(OperationMetrics::new);
        op_metrics.record_fallback();
    }

    pub fn get_metrics(&self, name: &str) -> Option<OperationMetrics> {
        let metrics = self.metrics.lock().unwrap();
        metrics.get(name).cloned()
    }

    pub fn report(&self) -> String {
        let metrics = self.metrics.lock().unwrap();
        let mut report = String::new();
        report.push_str("--- Performance Report ---\n");
        for (name, metrics) in metrics.iter() {
            report.push_str(&format!(
                "Operation: {}\n  Avg. Time: {:?}\n  Successes: {}\n  Failures: {}\n  Fallbacks: {}\n",
                name,
                metrics.average_execution_time(),
                metrics.success_count,
                metrics.failure_count,
                metrics.fallback_count
            ));
        }
        report
    }
}
