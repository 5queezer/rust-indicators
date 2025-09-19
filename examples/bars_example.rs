//! Example demonstrating López de Prado's information-driven bar types
//! 
//! This example shows how to use the different bar types for financial data sampling:
//! - Time bars: Fixed time intervals
//! - Volume bars: Fixed volume thresholds
//! - Dollar bars: Fixed dollar volume thresholds
//! - Volume Imbalance bars: Volume imbalance-driven sampling
//! - Tick Imbalance bars: Tick imbalance-driven sampling
//! - Run bars: Direction change-driven sampling

use rust_indicators::financial::bars::{BarBuilder, BarType, Tick};
use time::{OffsetDateTime, Duration};

fn main() {
    println!("López de Prado's Information-Driven Bar Types Example\n");

    // Create sample tick data
    let base_time = OffsetDateTime::now_utc();
    let ticks = vec![
        Tick { timestamp: base_time, price: 100.0, volume: 100 },
        Tick { timestamp: base_time + Duration::seconds(10), price: 101.0, volume: 150 },
        Tick { timestamp: base_time + Duration::seconds(20), price: 102.0, volume: 200 },
        Tick { timestamp: base_time + Duration::seconds(30), price: 101.5, volume: 120 },
        Tick { timestamp: base_time + Duration::seconds(40), price: 103.0, volume: 180 },
        Tick { timestamp: base_time + Duration::seconds(50), price: 102.5, volume: 90 },
        Tick { timestamp: base_time + Duration::seconds(60), price: 104.0, volume: 160 },
    ];

    // 1. Time Bars Example
    println!("=== Time Bars (30-second intervals) ===");
    let mut time_builder = BarBuilder::new(BarType::Time {
        frequency: Duration::seconds(30),
    });

    for tick in &ticks {
        if let Some(bar) = time_builder.process_tick(tick) {
            println!("Time Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
        }
    }

    // Force complete any remaining bar
    if let Some(bar) = time_builder.force_complete_bar() {
        println!("Final Time Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
    }

    // 2. Volume Bars Example
    println!("\n=== Volume Bars (300 volume threshold) ===");
    let mut volume_builder = BarBuilder::new(BarType::Volume { threshold: 300 });

    for tick in &ticks {
        if let Some(bar) = volume_builder.process_tick(tick) {
            println!("Volume Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
        }
    }

    if let Some(bar) = volume_builder.force_complete_bar() {
        println!("Final Volume Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
    }

    // 3. Dollar Bars Example
    println!("\n=== Dollar Bars ($30,000 threshold) ===");
    let mut dollar_builder = BarBuilder::new(BarType::Dollar { threshold: 30000.0 });

    for tick in &ticks {
        if let Some(bar) = dollar_builder.process_tick(tick) {
            println!("Dollar Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
        }
    }

    if let Some(bar) = dollar_builder.force_complete_bar() {
        println!("Final Dollar Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, VWAP={:.2}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap);
    }

    // 4. Tick Imbalance Bars Example
    println!("\n=== Tick Imbalance Bars (threshold=2) ===");
    let mut tick_imbalance_builder = BarBuilder::new(BarType::TickImbalance { threshold: 2 });

    for tick in &ticks {
        if let Some(bar) = tick_imbalance_builder.process_tick(tick) {
            println!("Tick Imbalance Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
        }
    }

    if let Some(bar) = tick_imbalance_builder.force_complete_bar() {
        println!("Final Tick Imbalance Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
    }

    // 5. Volume Imbalance Bars Example
    println!("\n=== Volume Imbalance Bars (threshold=100.0) ===");
    let mut volume_imbalance_builder = BarBuilder::new(BarType::VolumeImbalance { threshold: 100.0 });

    for tick in &ticks {
        if let Some(bar) = volume_imbalance_builder.process_tick(tick) {
            println!("Volume Imbalance Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
        }
    }

    if let Some(bar) = volume_imbalance_builder.force_complete_bar() {
        println!("Final Volume Imbalance Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
    }

    // 6. Run Bars Example
    println!("\n=== Run Bars (run_length=3) ===");
    let mut run_builder = BarBuilder::new(BarType::RunBars { run_length: 3 });

    for tick in &ticks {
        if let Some(bar) = run_builder.process_tick(tick) {
            println!("Run Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
        }
    }

    if let Some(bar) = run_builder.force_complete_bar() {
        println!("Final Run Bar: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Volume={}, Buy/Sell={}/{}",
            bar.open, bar.high, bar.low, bar.close, bar.volume, bar.buy_volume, bar.sell_volume);
    }

    println!("\n=== Summary ===");
    println!("This example demonstrates López de Prado's 6 information-driven bar types:");
    println!("1. Time bars: Sample at fixed time intervals");
    println!("2. Volume bars: Sample when volume threshold is reached");
    println!("3. Dollar bars: Sample when dollar volume threshold is reached");
    println!("4. Tick Imbalance bars: Sample when tick imbalance exceeds expectations");
    println!("5. Volume Imbalance bars: Sample when volume imbalance exceeds expectations");
    println!("6. Run bars: Sample when consecutive trades change direction");
    println!("\nThese bar types provide better synchronization with market information flow");
    println!("compared to traditional time-based sampling.");
}