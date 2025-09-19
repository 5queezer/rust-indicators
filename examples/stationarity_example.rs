//! Example demonstrating the stationarity testing suite
//! 
//! This example shows how to use the three key stationarity tests:
//! - ADF (Augmented Dickey-Fuller) test
//! - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
//! - Phillips-Perron test

use rust_indicators::financial::StationarityTests;
use ndarray::Array1;

fn main() {
    println!("=== Stationarity Testing Suite Example ===\n");
    
    // Example 1: White noise (should be stationary)
    println!("1. Testing white noise series (should be stationary):");
    let white_noise = Array1::from(vec![
        0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, 0.0,
        -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1,
        0.05, -0.15, 0.25, -0.05, 0.35, -0.25, 0.15, -0.35, 0.05, -0.05
    ]);
    
    let (adf_stat, adf_p, adf_stationary) = StationarityTests::adf_test(&white_noise);
    let (kpss_stat, kpss_p, kpss_stationary) = StationarityTests::kpss_test(&white_noise);
    let (pp_stat, pp_p, pp_stationary) = StationarityTests::phillips_perron_test(&white_noise);
    
    println!("  ADF Test:    statistic={:.4}, p-value={:.4}, stationary={}", 
             adf_stat, adf_p, adf_stationary);
    println!("  KPSS Test:   statistic={:.4}, p-value={:.4}, stationary={}", 
             kpss_stat, kpss_p, kpss_stationary);
    println!("  PP Test:     statistic={:.4}, p-value={:.4}, stationary={}", 
             pp_stat, pp_p, pp_stationary);
    
    // Example 2: Random walk (should be non-stationary)
    println!("\n2. Testing random walk series (should be non-stationary):");
    let mut random_walk = vec![0.0];
    for i in 1..30 {
        let increment = if i % 3 == 0 { 0.1 } else if i % 3 == 1 { -0.1 } else { 0.05 };
        random_walk.push(random_walk[i-1] + increment);
    }
    let random_walk = Array1::from(random_walk);
    
    let (adf_stat, adf_p, adf_stationary) = StationarityTests::adf_test(&random_walk);
    let (kpss_stat, kpss_p, kpss_stationary) = StationarityTests::kpss_test(&random_walk);
    let (pp_stat, pp_p, pp_stationary) = StationarityTests::phillips_perron_test(&random_walk);
    
    println!("  ADF Test:    statistic={:.4}, p-value={:.4}, stationary={}", 
             adf_stat, adf_p, adf_stationary);
    println!("  KPSS Test:   statistic={:.4}, p-value={:.4}, stationary={}", 
             kpss_stat, kpss_p, kpss_stationary);
    println!("  PP Test:     statistic={:.4}, p-value={:.4}, stationary={}", 
             pp_stat, pp_p, pp_stationary);
    
    // Example 3: Trend-stationary series
    println!("\n3. Testing trend-stationary series:");
    let trend_stationary: Array1<f64> = (0..25)
        .map(|i| i as f64 * 0.1 + (i as f64 * 0.2).sin() * 0.3)
        .collect::<Vec<_>>()
        .into();
    
    let (adf_stat, adf_p, adf_stationary) = StationarityTests::adf_test(&trend_stationary);
    let (kpss_stat, kpss_p, kpss_stationary) = StationarityTests::kpss_test(&trend_stationary);
    let (pp_stat, pp_p, pp_stationary) = StationarityTests::phillips_perron_test(&trend_stationary);
    
    println!("  ADF Test:    statistic={:.4}, p-value={:.4}, stationary={}", 
             adf_stat, adf_p, adf_stationary);
    println!("  KPSS Test:   statistic={:.4}, p-value={:.4}, stationary={}", 
             kpss_stat, kpss_p, kpss_stationary);
    println!("  PP Test:     statistic={:.4}, p-value={:.4}, stationary={}", 
             pp_stat, pp_p, pp_stationary);
    
    println!("\n=== Test Interpretation ===");
    println!("ADF & PP Tests: Null hypothesis = unit root (non-stationary)");
    println!("  - Reject null (p < 0.05) → series is stationary");
    println!("  - Fail to reject null (p ≥ 0.05) → series has unit root");
    println!();
    println!("KPSS Test: Null hypothesis = trend stationary");
    println!("  - Reject null (p < 0.05) → series has unit root");
    println!("  - Fail to reject null (p ≥ 0.05) → series is stationary");
    println!();
    println!("For robust conclusions, use all three tests together!");
}