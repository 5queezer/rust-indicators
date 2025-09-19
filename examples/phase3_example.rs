//! Phase 3 Implementation Example
//! 
//! This example demonstrates the advanced ML components introduced in Phase 3:
//! 1. Side-aware Triple Barrier Labeling (Long/Short positions)
//! 2. Meta-Labeling strategy with volatility adjustment
//! 3. Path-dependent barrier logic
//! 4. Parallel processing capabilities
//! 5. Complete workflow integration
//!
//! Based on López de Prado's "Advances in Financial Machine Learning" triple barrier
//! method and meta-labeling strategies for sophisticated ML-driven trading systems.

use rust_indicators::ml::components::{
    TradingSide, TripleBarrierLabeler, MetaLabeler, ConfidencePredictor
};
use rand::prelude::*;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    println!("=== Phase 3 ML Components Example ===\n");
    println!("Demonstrating López de Prado's Triple Barrier Method and Meta-Labeling\n");

    // Generate synthetic financial data for demonstration
    let (prices, volatility, features) = generate_synthetic_data(1000);
    
    println!("Generated {} price points with volatility estimates", prices.len());
    println!("Price range: {:.2} - {:.2}", 
        prices.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        prices.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("Average volatility: {:.4}\n", volatility.iter().sum::<f32>() / volatility.len() as f32);

    // === 1. SIDE-AWARE TRIPLE BARRIER LABELING ===
    println!("=== 1. Side-Aware Triple Barrier Labeling ===");
    
    // Demonstrate all 8 barrier configurations
    demonstrate_barrier_configurations(&prices, &volatility);
    
    // === 2. PATH-DEPENDENT BARRIER LOGIC ===
    println!("\n=== 2. Path-Dependent Barrier Logic ===");
    demonstrate_path_dependent_logic(&prices, &volatility);
    
    // === 3. META-LABELING WORKFLOW ===
    println!("\n=== 3. Meta-Labeling Workflow ===");
    demonstrate_meta_labeling(&prices, &volatility, &features);
    
    // === 4. PARALLEL PROCESSING PERFORMANCE ===
    println!("\n=== 4. Parallel Processing Performance ===");
    demonstrate_parallel_performance(&prices, &volatility);
    
    // === 5. COMPLETE INTEGRATION WORKFLOW ===
    println!("\n=== 5. Complete Integration Workflow ===");
    demonstrate_complete_workflow(&prices, &volatility, &features);
    
    println!("\n=== Summary ===");
    println!("Phase 3 components provide:");
    println!("• Side-aware labeling for Long/Short strategies");
    println!("• Path-dependent barrier logic for realistic trading");
    println!("• Meta-labeling for precision improvement");
    println!("• Parallel processing for performance");
    println!("• Complete ML workflow integration");
}

/// Generate synthetic financial data for demonstration
fn generate_synthetic_data(n: usize) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let mut rng = thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut volatility = Vec::with_capacity(n);
    let mut features = Vec::with_capacity(n);
    
    let mut price = 100.0;
    let base_vol = 0.02;
    
    for i in 0..n {
        // Generate price with mean reversion and volatility clustering
        let vol = base_vol * (1.0 + 0.5 * (i as f32 / 100.0).sin()) * rng.gen_range(0.5..2.0);
        let return_val = rng.gen_range(-3.0..3.0) * vol;
        price *= 1.0 + return_val;
        
        prices.push(price);
        volatility.push(vol);
        
        // Generate 5 synthetic features (RSI-like, MA ratio, momentum, volume, sentiment)
        let feature_vec = vec![
            rng.gen_range(-1.0..1.0),  // RSI-like oscillator
            rng.gen_range(-0.5..0.5),  // MA ratio
            return_val * 10.0,         // Momentum (scaled return)
            rng.gen_range(0.0..2.0),   // Volume indicator
            rng.gen_range(-1.0..1.0),  // Sentiment indicator
        ];
        features.push(feature_vec);
    }
    
    (prices, volatility, features)
}

/// Demonstrate all 8 barrier configurations (Long/Short × 4 barrier outcomes)
fn demonstrate_barrier_configurations(prices: &[f32], volatility: &[f32]) {
    println!("Testing all 8 barrier configurations:");
    println!("Long positions: Upper=Profit, Lower=Loss, Vertical=Time, None=No signal");
    println!("Short positions: Upper=Loss, Lower=Profit, Vertical=Time, None=No signal\n");
    
    // Long position labeling
    let long_labeler = TripleBarrierLabeler::new(2.0, 1.5, 20)
        .with_side(TradingSide::Long);
    
    // Short position labeling  
    let short_labeler = TripleBarrierLabeler::new(2.0, 1.5, 20)
        .with_side(TradingSide::Short);
    
    // Test on a subset for clear demonstration
    let test_prices = &prices[0..100];
    let test_vols = &volatility[0..100];
    
    // Simulate label generation (normally would use Python interface)
    let long_results = simulate_labeling(&long_labeler, test_prices, test_vols, TradingSide::Long);
    let short_results = simulate_labeling(&short_labeler, test_prices, test_vols, TradingSide::Short);
    
    println!("Long Position Results:");
    print_label_distribution(&long_results);
    
    println!("\nShort Position Results:");
    print_label_distribution(&short_results);
    
    // Show specific examples
    println!("\nExample barrier touches:");
    for i in 0..5 {
        if i < test_prices.len() - 20 {
            let entry = test_prices[i];
            let vol = test_vols[i];
            let upper = entry * (1.0 + 2.0 * vol);
            let lower = entry * (1.0 - 1.5 * vol);
            
            println!("Entry {}: Price={:.2}, Upper={:.2} (+{:.1}%), Lower={:.2} (-{:.1}%)", 
                i, entry, upper, (upper/entry - 1.0) * 100.0, lower, (1.0 - lower/entry) * 100.0);
        }
    }
}

/// Simulate labeling process (would normally use Python interface)
fn simulate_labeling(labeler: &TripleBarrierLabeler, prices: &[f32], volatility: &[f32], side: TradingSide) -> Vec<i32> {
    let mut labels = Vec::new();
    
    for i in 0..(prices.len().saturating_sub(labeler.default_max_hold)) {
        let entry = prices[i];
        let vol = volatility[i];
        let upper_barrier = entry * (1.0 + labeler.default_profit_mult * vol);
        let lower_barrier = entry * (1.0 - labeler.default_stop_mult * vol);
        
        // Check barriers in next max_hold periods
        let mut label = 1; // Default hold
        let end_idx = (i + labeler.default_max_hold).min(prices.len() - 1);
        
        for j in (i + 1)..=end_idx {
            let price = prices[j];
            
            if price >= upper_barrier {
                // Upper barrier hit
                label = match side {
                    TradingSide::Long => 1,   // Profit
                    TradingSide::Short => -1, // Loss
                };
                break;
            } else if price <= lower_barrier {
                // Lower barrier hit
                label = match side {
                    TradingSide::Long => -1,  // Loss
                    TradingSide::Short => 1,  // Profit
                };
                break;
            }
        }
        
        // If no barrier hit, use final return
        if label == 1 && end_idx > i {
            let final_return = (prices[end_idx] / entry) - 1.0;
            label = match side {
                TradingSide::Long => {
                    if final_return > 0.002 { 1 } else if final_return < -0.002 { -1 } else { 0 }
                },
                TradingSide::Short => {
                    if final_return < -0.002 { 1 } else if final_return > 0.002 { -1 } else { 0 }
                },
            };
        }
        
        labels.push(label);
    }
    
    labels
}

/// Print label distribution statistics
fn print_label_distribution(labels: &[i32]) {
    let mut counts = [0; 3]; // [-1, 0, 1]
    for &label in labels {
        match label {
            -1 => counts[0] += 1,
            0 => counts[1] += 1,
            1 => counts[2] += 1,
            _ => {}
        }
    }
    
    let total = labels.len() as f32;
    println!("  Loss (-1): {} ({:.1}%)", counts[0], counts[0] as f32 / total * 100.0);
    println!("  Hold (0):  {} ({:.1}%)", counts[1], counts[1] as f32 / total * 100.0);
    println!("  Profit (1): {} ({:.1}%)", counts[2], counts[2] as f32 / total * 100.0);
}

/// Demonstrate path-dependent barrier logic
fn demonstrate_path_dependent_logic(prices: &[f32], volatility: &[f32]) {
    println!("Path-dependent logic ensures barriers are checked in chronological order:");
    println!("This prevents look-ahead bias and reflects realistic trading conditions.\n");
    
    // Show a specific example of path dependency
    let start_idx = 50;
    let entry_price = prices[start_idx];
    let vol = volatility[start_idx];
    let upper_barrier = entry_price * (1.0 + 2.0 * vol);
    let lower_barrier = entry_price * (1.0 - 1.5 * vol);
    let max_hold = 20;
    
    println!("Example path analysis (Entry at index {}):", start_idx);
    println!("Entry Price: {:.2}", entry_price);
    println!("Upper Barrier: {:.2} (+{:.1}%)", upper_barrier, (upper_barrier/entry_price - 1.0) * 100.0);
    println!("Lower Barrier: {:.2} (-{:.1}%)", lower_barrier, (1.0 - lower_barrier/entry_price) * 100.0);
    println!("Max Hold: {} periods\n", max_hold);
    
    println!("Price path:");
    let end_idx = (start_idx + max_hold).min(prices.len() - 1);
    for i in start_idx..=end_idx {
        let price = prices[i];
        let period = i - start_idx;
        let return_pct = (price / entry_price - 1.0) * 100.0;
        
        let status = if price >= upper_barrier {
            "UPPER HIT! 🎯"
        } else if price <= lower_barrier {
            "LOWER HIT! 🛑"
        } else if period == max_hold {
            "TIME EXPIRED ⏰"
        } else {
            "Monitoring..."
        };
        
        println!("  Period {}: {:.2} ({:+.1}%) - {}", period, price, return_pct, status);
        
        // Stop at first barrier hit
        if price >= upper_barrier || price <= lower_barrier {
            break;
        }
    }
}

/// Demonstrate meta-labeling workflow
fn demonstrate_meta_labeling(_prices: &[f32], volatility: &[f32], features: &[Vec<f32>]) {
    println!("Meta-labeling combines a primary model (white-box) with ML filtering:");
    println!("1. Primary model generates trading signals");
    println!("2. Meta-model decides whether to act on each signal");
    println!("3. Result: Improved precision through false positive filtering\n");
    
    // Create primary predictor (simulated fundamental model)
    let mut primary_predictor = ConfidencePredictor::new(0.6, 5);
    
    // Simulate trained weights (e.g., from fundamental analysis)
    let primary_weights = vec![0.3, -0.2, 0.5, 0.1, -0.4]; // RSI, MA, Momentum, Volume, Sentiment
    let importance = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    primary_predictor.set_weights(primary_weights, importance).unwrap();
    
    // Create meta-labeler
    let meta_labeler = MetaLabeler::new(primary_predictor, 0.5, 0.2);
    
    println!("Primary Model Configuration:");
    println!("  Features: RSI, MA_Ratio, Momentum, Volume, Sentiment");
    println!("  Weights: [0.3, -0.2, 0.5, 0.1, -0.4]");
    println!("  Confidence Threshold: 0.5");
    println!("  Volatility Adjustment: 0.2\n");
    
    // Test meta-labeling on sample data
    println!("Meta-labeling results (first 10 samples):");
    println!("Sample | Primary Pred | Confidence | Volatility | Meta Decision | Reason");
    println!("-------|--------------|------------|------------|---------------|--------");
    
    for i in 0..10.min(features.len()) {
        if let (Ok((pred, conf)), vol) = (
            meta_labeler.primary_predictor.predict_sample(&features[i]),
            volatility[i]
        ) {
            let meta_decision = meta_labeler.meta_predict(&features[i], vol).unwrap_or(0);
            let adjusted_threshold = 0.5 * (1.0 + 0.2 * vol);
            
            let reason = if conf < adjusted_threshold {
                "Low confidence"
            } else if pred == 1 {
                "Hold signal"
            } else if meta_decision == 1 {
                "Meta approved"
            } else {
                "Meta filtered"
            };
            
            let pred_str = match pred {
                0 => "Sell",
                1 => "Hold", 
                2 => "Buy",
                _ => "Unknown"
            };
            
            println!("  {:2}   |     {:4}     |   {:.3}    |   {:.3}    |      {}       | {}",
                i, pred_str, conf, vol, meta_decision, reason);
        }
    }
    
    // Show meta-labeling statistics
    let mut primary_signals = 0;
    let mut meta_approved = 0;
    
    for i in 0..100.min(features.len()) {
        if let Ok((pred, conf)) = meta_labeler.primary_predictor.predict_sample(&features[i]) {
            if pred != 1 && conf > 0.3 { // Non-hold signals with some confidence
                primary_signals += 1;
                if let Ok(meta_decision) = meta_labeler.meta_predict(&features[i], volatility[i]) {
                    if meta_decision == 1 {
                        meta_approved += 1;
                    }
                }
            }
        }
    }
    
    println!("\nMeta-labeling Statistics (first 100 samples):");
    println!("  Primary signals generated: {}", primary_signals);
    println!("  Meta-approved signals: {}", meta_approved);
    if primary_signals > 0 {
        println!("  Approval rate: {:.1}%", meta_approved as f32 / primary_signals as f32 * 100.0);
    }
    println!("  → Meta-labeling filters out {:.1}% of primary signals", 
        (1.0 - meta_approved as f32 / primary_signals.max(1) as f32) * 100.0);
}

/// Demonstrate parallel processing performance
fn demonstrate_parallel_performance(prices: &[f32], volatility: &[f32]) {
    println!("Comparing sequential vs parallel processing performance:");
    
    let labeler = TripleBarrierLabeler::new(2.0, 1.5, 20);
    let test_size = 500.min(prices.len() - 20);
    
    // Sequential processing
    let start = Instant::now();
    let sequential_results = simulate_labeling_sequential(&labeler, &prices[..test_size], &volatility[..test_size]);
    let sequential_time = start.elapsed();
    
    // Parallel processing (simulated)
    let start = Instant::now();
    let parallel_results = simulate_labeling_parallel(&labeler, &prices[..test_size], &volatility[..test_size]);
    let parallel_time = start.elapsed();
    
    println!("Performance Results:");
    println!("  Dataset size: {} samples", test_size);
    println!("  Sequential time: {:?}", sequential_time);
    println!("  Parallel time: {:?}", parallel_time);
    
    if sequential_time > parallel_time {
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("  Speedup: {:.2}x faster with parallel processing", speedup);
    } else {
        println!("  Note: Overhead dominates for small datasets");
    }
    
    // Verify results are identical
    let identical = sequential_results.len() == parallel_results.len() &&
        sequential_results.iter().zip(&parallel_results).all(|(a, b)| a == b);
    
    println!("  Results identical: {}", if identical { "✓" } else { "✗" });
    
    println!("\nParallel processing benefits:");
    println!("  • Scales with CPU cores");
    println!("  • Ideal for large datasets (>1000 samples)");
    println!("  • Maintains deterministic results");
    println!("  • Built-in with Rayon for zero-cost abstraction");
}

/// Sequential labeling simulation
fn simulate_labeling_sequential(labeler: &TripleBarrierLabeler, prices: &[f32], volatility: &[f32]) -> Vec<i32> {
    (0..prices.len().saturating_sub(labeler.default_max_hold))
        .map(|i| {
            let entry = prices[i];
            let vol = volatility[i];
            let upper = entry * (1.0 + labeler.default_profit_mult * vol);
            let lower = entry * (1.0 - labeler.default_stop_mult * vol);
            
            // Simplified barrier check
            let end_idx = (i + labeler.default_max_hold).min(prices.len() - 1);
            for j in (i + 1)..=end_idx {
                if prices[j] >= upper { return 1; }
                if prices[j] <= lower { return -1; }
            }
            0 // No barrier hit
        })
        .collect()
}

/// Parallel labeling simulation
fn simulate_labeling_parallel(labeler: &TripleBarrierLabeler, prices: &[f32], volatility: &[f32]) -> Vec<i32> {
    (0..prices.len().saturating_sub(labeler.default_max_hold))
        .into_par_iter()
        .map(|i| {
            let entry = prices[i];
            let vol = volatility[i];
            let upper = entry * (1.0 + labeler.default_profit_mult * vol);
            let lower = entry * (1.0 - labeler.default_stop_mult * vol);
            
            // Simplified barrier check
            let end_idx = (i + labeler.default_max_hold).min(prices.len() - 1);
            for j in (i + 1)..=end_idx {
                if prices[j] >= upper { return 1; }
                if prices[j] <= lower { return -1; }
            }
            0 // No barrier hit
        })
        .collect()
}

/// Demonstrate complete workflow integration
fn demonstrate_complete_workflow(prices: &[f32], volatility: &[f32], features: &[Vec<f32>]) {
    println!("Complete ML workflow: Data → Labels → Training → Prediction → Meta-filtering");
    println!("This demonstrates the full pipeline from raw data to trading decisions.\n");
    
    let workflow_size = 200.min(prices.len() - 20);
    
    // Step 1: Generate labels using triple barrier method
    println!("Step 1: Label Generation");
    let long_labeler = TripleBarrierLabeler::new(2.0, 1.5, 15).with_side(TradingSide::Long);
    let labels = simulate_labeling(&long_labeler, &prices[..workflow_size], &volatility[..workflow_size], TradingSide::Long);
    
    let label_stats = count_labels(&labels);
    println!("  Generated {} labels: {} profitable, {} unprofitable, {} neutral", 
        labels.len(), label_stats[2], label_stats[0], label_stats[1]);
    
    // Step 2: Train primary model (simulated)
    println!("\nStep 2: Primary Model Training");
    let mut primary_predictor = ConfidencePredictor::new(0.6, 5);
    
    // Simulate training by setting reasonable weights
    let trained_weights = vec![0.4, -0.3, 0.6, 0.2, -0.1];
    let importance = vec![0.8, 0.9, 1.0, 0.6, 0.7];
    primary_predictor.set_weights(trained_weights, importance).unwrap();
    println!("  Primary model trained with 5 features");
    
    // Step 3: Generate predictions
    println!("\nStep 3: Primary Predictions");
    let mut predictions = Vec::new();
    let mut confidences = Vec::new();
    
    for i in 0..50.min(features.len()) {
        if let Ok((pred, conf)) = primary_predictor.predict_sample(&features[i]) {
            predictions.push(pred);
            confidences.push(conf);
        }
    }
    
    let pred_stats = count_predictions(&predictions);
    println!("  Generated {} predictions: {} buy, {} hold, {} sell", 
        predictions.len(), pred_stats[2], pred_stats[1], pred_stats[0]);
    
    // Step 4: Meta-labeling filter
    println!("\nStep 4: Meta-labeling Filter");
    let meta_labeler = MetaLabeler::new(primary_predictor, 0.5, 0.15);
    
    let mut final_decisions = Vec::new();
    for i in 0..predictions.len().min(features.len()).min(volatility.len()) {
        if let Ok(decision) = meta_labeler.meta_predict(&features[i], volatility[i]) {
            final_decisions.push(decision);
        }
    }
    
    let approved = final_decisions.iter().sum::<i32>();
    println!("  Meta-filter approved {} out of {} signals ({:.1}%)", 
        approved, final_decisions.len(), 
        approved as f32 / final_decisions.len().max(1) as f32 * 100.0);
    
    // Step 5: Performance summary
    println!("\nStep 5: Workflow Summary");
    println!("  Data points processed: {}", workflow_size);
    println!("  Labels generated: {}", labels.len());
    println!("  Primary predictions: {}", predictions.len());
    println!("  Final trading signals: {}", approved);
    println!("  Signal reduction: {:.1}%", 
        (1.0 - approved as f32 / predictions.len().max(1) as f32) * 100.0);
    
    println!("\nWorkflow Benefits:");
    println!("  ✓ Systematic label generation with risk management");
    println!("  ✓ Side-aware labeling for directional strategies");
    println!("  ✓ Meta-labeling for precision improvement");
    println!("  ✓ Volatility-adjusted thresholds");
    println!("  ✓ Parallel processing for scalability");
    println!("  ✓ Complete integration from data to decisions");
}

/// Count label distribution
fn count_labels(labels: &[i32]) -> [usize; 3] {
    let mut counts = [0; 3]; // [-1, 0, 1]
    for &label in labels {
        match label {
            -1 => counts[0] += 1,
            0 => counts[1] += 1,
            1 => counts[2] += 1,
            _ => {}
        }
    }
    counts
}

/// Count prediction distribution
fn count_predictions(predictions: &[i32]) -> [usize; 3] {
    let mut counts = [0; 3]; // [0, 1, 2] -> [sell, hold, buy]
    for &pred in predictions {
        match pred {
            0 => counts[0] += 1,
            1 => counts[1] += 1,
            2 => counts[2] += 1,
            _ => {}
        }
    }
    counts
}