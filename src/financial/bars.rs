//! Information-driven bar structures for financial data sampling
//!
//! This module implements López de Prado's advanced bar types from "Advances in Financial Machine Learning"
//! for information-driven sampling. These bar types sample more frequently when new information arrives
//! to the market, providing better synchronization with informed trading activity.
//!
//! # Bar Types
//!
//! 1. **Time bars**: Sample at fixed time intervals
//! 2. **Volume bars**: Sample when volume threshold is reached
//! 3. **Dollar bars**: Sample when dollar volume threshold is reached
//! 4. **Volume Imbalance bars**: Sample when volume imbalance exceeds expectations
//! 5. **Tick Imbalance bars**: Sample when tick imbalance exceeds expectations
//! 6. **Run bars**: Sample when consecutive trades change direction
//!
//! # Example
//!
//! ```rust
//! use rust_indicators::financial::bars::{BarBuilder, BarType, Tick};
//! use time::{OffsetDateTime, Duration};
//!
//! // Create a volume bar builder with 1000 volume threshold
//! let mut builder = BarBuilder::new(BarType::Volume { threshold: 1000 });
//!
//! // Process ticks
//! let tick = Tick {
//!     timestamp: OffsetDateTime::now_utc(),
//!     price: 100.0,
//!     volume: 500,
//! };
//!
//! if let Some(bar) = builder.process_tick(&tick) {
//!     println!("New bar created: {:?}", bar);
//! }
//! ```

use std::collections::VecDeque;
use time::{Duration, OffsetDateTime};

/// Bar type enumeration defining the 6 information-driven sampling methods
#[derive(Debug, Clone, PartialEq)]
pub enum BarType {
    /// Time-based bars sampled at fixed intervals
    Time { frequency: Duration },
    /// Volume-based bars sampled when volume threshold is reached
    Volume { threshold: u64 },
    /// Dollar volume-based bars sampled when dollar threshold is reached
    Dollar { threshold: f64 },
    /// Volume imbalance bars sampled when volume imbalance exceeds expectations
    VolumeImbalance { threshold: f64 },
    /// Tick imbalance bars sampled when tick imbalance exceeds expectations
    TickImbalance { threshold: i32 },
    /// Run bars sampled when consecutive trades change direction
    RunBars { run_length: usize },
}

/// A complete bar with OHLCV data plus additional microstructural information
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    /// Timestamp when the bar was completed
    pub timestamp: OffsetDateTime,
    /// Opening price of the bar
    pub open: f64,
    /// Highest price in the bar
    pub high: f64,
    /// Lowest price in the bar
    pub low: f64,
    /// Closing price of the bar
    pub close: f64,
    /// Total volume in the bar
    pub volume: u64,
    /// Volume-weighted average price
    pub vwap: f64,
    /// Volume attributed to buy trades
    pub buy_volume: u64,
    /// Volume attributed to sell trades
    pub sell_volume: u64,
}

/// A single tick representing a trade or quote update
#[derive(Debug, Clone, PartialEq)]
pub struct Tick {
    /// Timestamp of the tick
    pub timestamp: OffsetDateTime,
    /// Price of the trade/quote
    pub price: f64,
    /// Volume of the trade (0 for quotes)
    pub volume: u64,
}

/// Tracks imbalances for volume and tick imbalance bars
#[derive(Debug, Clone)]
pub struct ImbalanceTracker {
    /// Current tick imbalance (sum of signed ticks)
    pub tick_imbalance: i32,
    /// Current volume imbalance (sum of signed volumes)
    pub volume_imbalance: f64,
    /// Expected tick imbalance based on historical data
    pub expected_tick_imbalance: f64,
    /// Expected volume imbalance based on historical data
    pub expected_volume_imbalance: f64,
    /// Historical tick signs for expectation calculation
    tick_history: VecDeque<i8>,
    /// Historical volume imbalances for expectation calculation
    volume_history: VecDeque<f64>,
    /// Maximum history length for expectation calculation
    max_history: usize,
}

impl ImbalanceTracker {
    /// Create a new imbalance tracker
    pub fn new(max_history: usize) -> Self {
        Self {
            tick_imbalance: 0,
            volume_imbalance: 0.0,
            expected_tick_imbalance: 0.0,
            expected_volume_imbalance: 0.0,
            tick_history: VecDeque::with_capacity(max_history),
            volume_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Update imbalances with a new tick
    pub fn update(&mut self, tick_sign: i8, volume: u64, _price: f64) {
        // Update current imbalances
        self.tick_imbalance += tick_sign as i32;
        let signed_volume = tick_sign as f64 * volume as f64;
        self.volume_imbalance += signed_volume;

        // Update history for expectation calculation
        self.tick_history.push_back(tick_sign);
        self.volume_history.push_back(signed_volume);

        // Maintain history size
        if self.tick_history.len() > self.max_history {
            self.tick_history.pop_front();
        }
        if self.volume_history.len() > self.max_history {
            self.volume_history.pop_front();
        }

        // Update expectations based on historical averages
        self.update_expectations();
    }

    /// Reset imbalances for a new bar
    pub fn reset(&mut self) {
        self.tick_imbalance = 0;
        self.volume_imbalance = 0.0;
    }

    /// Check if tick imbalance exceeds threshold based on López de Prado's methodology
    pub fn tick_imbalance_exceeded(&self, threshold: i32) -> bool {
        // For simple implementation, use absolute threshold
        // In practice, this would use |θ_T| >= |E[θ_T]| where E[θ_T] = E[T] * (2P[b=1] - 1)
        self.tick_imbalance.abs() > threshold
    }

    /// Check if volume imbalance exceeds threshold based on López de Prado's methodology
    pub fn volume_imbalance_exceeded(&self, threshold: f64) -> bool {
        // For simple implementation, use absolute threshold
        // In practice, this would use |θ_T| >= |E[θ_T]| where E[θ_T] = E[T] * (2P[b=1] - 1) * avg_volume
        self.volume_imbalance.abs() > threshold
    }

    /// Update expectations based on historical data
    fn update_expectations(&mut self) {
        if !self.tick_history.is_empty() {
            let tick_sum: i32 = self.tick_history.iter().map(|&x| x as i32).sum();
            self.expected_tick_imbalance = tick_sum as f64 / self.tick_history.len() as f64;
        }

        if !self.volume_history.is_empty() {
            self.expected_volume_imbalance =
                self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
        }
    }
}

/// Builder for streaming bar construction using various sampling methods
pub struct BarBuilder {
    /// Type of bar being constructed
    bar_type: BarType,
    /// Current bar being built (None if no bar in progress)
    current_bar: Option<BarInProgress>,
    /// Imbalance tracker for imbalance-based bars
    imbalance_tracker: ImbalanceTracker,
    /// Previous tick for tick rule calculation
    previous_tick: Option<Tick>,
    /// Current run state for run bars
    run_state: RunState,
}

/// Internal structure for tracking a bar in progress
#[derive(Debug, Clone)]
struct BarInProgress {
    start_time: OffsetDateTime,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
    dollar_volume: f64,
    buy_volume: u64,
    sell_volume: u64,
    tick_count: u32,
}

/// State tracking for run bars
#[derive(Debug, Clone)]
struct RunState {
    current_direction: Option<i8>, // 1 for up, -1 for down, None for no direction
    run_length: usize,
    consecutive_ticks: usize,
}

impl BarBuilder {
    /// Create a new bar builder with the specified bar type
    pub fn new(bar_type: BarType) -> Self {
        Self {
            bar_type,
            current_bar: None,
            imbalance_tracker: ImbalanceTracker::new(1000), // Default history size
            previous_tick: None,
            run_state: RunState {
                current_direction: None,
                run_length: 0,
                consecutive_ticks: 0,
            },
        }
    }

    /// Process a new tick and potentially return a completed bar
    ///
    /// This is the main entry point for streaming bar construction.
    /// Returns Some(Bar) when a bar is completed according to the sampling rules.
    pub fn process_tick(&mut self, tick: &Tick) -> Option<Bar> {
        // Calculate tick sign using the tick rule
        let tick_sign = self.calculate_tick_sign(tick);

        // Initialize or update current bar
        if self.current_bar.is_none() {
            self.start_new_bar(tick);
        }

        // Update current bar with tick
        self.update_current_bar_with_tick(tick, tick_sign);

        // Check if bar should be completed based on bar type
        if self.should_complete_bar(tick, tick_sign) {
            self.complete_current_bar()
        } else {
            None
        }
    }

    /// Calculate tick sign using the tick rule
    ///
    /// Returns:
    /// - 1 for uptick (price increase)
    /// - -1 for downtick (price decrease)  
    /// - 0 for no change (uses previous tick sign if available)
    fn calculate_tick_sign(&mut self, tick: &Tick) -> i8 {
        let sign = if let Some(ref prev) = self.previous_tick {
            if tick.price > prev.price {
                1
            } else if tick.price < prev.price {
                -1
            } else {
                0 // No change - will use previous direction
            }
        } else {
            0 // First tick
        };

        self.previous_tick = Some(tick.clone());
        sign
    }

    /// Start a new bar with the given tick
    fn start_new_bar(&mut self, tick: &Tick) {
        self.current_bar = Some(BarInProgress {
            start_time: tick.timestamp,
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            volume: 0,
            dollar_volume: 0.0,
            buy_volume: 0,
            sell_volume: 0,
            tick_count: 0,
        });

        // Reset trackers for new bar
        self.imbalance_tracker.reset();
        self.run_state.consecutive_ticks = 0;
    }

    /// Update the current bar with a new tick
    fn update_current_bar_with_tick(&mut self, tick: &Tick, tick_sign: i8) {
        if let Some(ref mut bar) = self.current_bar {
            // Update OHLC
            bar.high = bar.high.max(tick.price);
            bar.low = bar.low.min(tick.price);
            bar.close = tick.price;

            // Update volume and dollar volume
            bar.volume += tick.volume;
            bar.dollar_volume += tick.price * tick.volume as f64;
            bar.tick_count += 1;

            // Classify trade as buy or sell and update volumes
            if tick_sign > 0 {
                bar.buy_volume += tick.volume;
            } else if tick_sign < 0 {
                bar.sell_volume += tick.volume;
            } else {
                // For zero tick sign, split volume evenly or use previous direction
                bar.buy_volume += tick.volume / 2;
                bar.sell_volume += tick.volume / 2;
            }
        }

        // Update imbalance tracker
        self.imbalance_tracker
            .update(tick_sign, tick.volume, tick.price);

        // Update run state
        self.update_run_state(tick_sign);
    }

    /// Update run state for run bars
    fn update_run_state(&mut self, tick_sign: i8) {
        if tick_sign == 0 {
            return; // Ignore neutral ticks for runs
        }

        match self.run_state.current_direction {
            None => {
                // Start new run
                self.run_state.current_direction = Some(tick_sign);
                self.run_state.consecutive_ticks = 1;
                self.run_state.run_length = 0; // No completed run yet
            }
            Some(current_dir) if current_dir == tick_sign => {
                // Continue current run
                self.run_state.consecutive_ticks += 1;
            }
            Some(_) => {
                // Direction changed - complete the previous run
                self.run_state.run_length = self.run_state.consecutive_ticks;
                // Start new run with current tick
                self.run_state.current_direction = Some(tick_sign);
                self.run_state.consecutive_ticks = 1;
            }
        }
    }

    /// Check if the current bar should be completed based on bar type
    fn should_complete_bar(&self, tick: &Tick, _tick_sign: i8) -> bool {
        if let Some(ref bar) = self.current_bar {
            match &self.bar_type {
                BarType::Time { frequency } => tick.timestamp >= bar.start_time + *frequency,
                BarType::Volume { threshold } => bar.volume >= *threshold,
                BarType::Dollar { threshold } => bar.dollar_volume >= *threshold,
                BarType::VolumeImbalance { threshold } => {
                    self.imbalance_tracker.volume_imbalance_exceeded(*threshold)
                }
                BarType::TickImbalance { threshold } => {
                    self.imbalance_tracker.tick_imbalance_exceeded(*threshold)
                }
                BarType::RunBars { run_length } => self.run_state.run_length >= *run_length,
            }
        } else {
            false
        }
    }

    /// Complete the current bar and return it
    fn complete_current_bar(&mut self) -> Option<Bar> {
        if let Some(bar_progress) = self.current_bar.take() {
            let vwap = if bar_progress.volume > 0 {
                bar_progress.dollar_volume / bar_progress.volume as f64
            } else {
                bar_progress.close
            };

            Some(Bar {
                timestamp: bar_progress.start_time,
                open: bar_progress.open,
                high: bar_progress.high,
                low: bar_progress.low,
                close: bar_progress.close,
                volume: bar_progress.volume,
                vwap,
                buy_volume: bar_progress.buy_volume,
                sell_volume: bar_progress.sell_volume,
            })
        } else {
            None
        }
    }

    /// Get the current bar type
    pub fn bar_type(&self) -> &BarType {
        &self.bar_type
    }

    /// Check if a bar is currently in progress
    pub fn has_current_bar(&self) -> bool {
        self.current_bar.is_some()
    }

    /// Force completion of the current bar (useful for end-of-session)
    pub fn force_complete_bar(&mut self) -> Option<Bar> {
        self.complete_current_bar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Duration;

    fn create_test_tick(timestamp: OffsetDateTime, price: f64, volume: u64) -> Tick {
        Tick {
            timestamp,
            price,
            volume,
        }
    }

    #[test]
    fn test_time_bars() {
        let mut builder = BarBuilder::new(BarType::Time {
            frequency: Duration::minutes(1),
        });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100);
        let tick2 = create_test_tick(base_time + Duration::seconds(30), 101.0, 200);
        let tick3 = create_test_tick(base_time + Duration::minutes(1), 102.0, 150);

        // First two ticks should not complete bar
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Third tick should complete bar due to time threshold
        let bar = builder.process_tick(&tick3).unwrap();
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 102.0); // Close with the triggering tick
        assert_eq!(bar.volume, 450); // All three ticks included
    }

    #[test]
    fn test_volume_bars() {
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 250 });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100);
        let tick2 = create_test_tick(base_time + Duration::seconds(1), 101.0, 100);
        let tick3 = create_test_tick(base_time + Duration::seconds(2), 102.0, 100);

        // First two ticks should not complete bar
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Third tick should complete bar due to volume threshold
        let bar = builder.process_tick(&tick3).unwrap();
        assert_eq!(bar.volume, 300);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 102.0);
    }

    #[test]
    fn test_dollar_bars() {
        let mut builder = BarBuilder::new(BarType::Dollar { threshold: 20000.0 });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100); // $10,000
        let tick2 = create_test_tick(base_time + Duration::seconds(1), 101.0, 50); // $5,050
        let tick3 = create_test_tick(base_time + Duration::seconds(2), 102.0, 50); // $5,100

        // First two ticks should not complete bar
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Third tick should complete bar due to dollar volume threshold
        let bar = builder.process_tick(&tick3).unwrap();
        assert_eq!(bar.volume, 200);
        // Total dollar volume: 10000 + 5050 + 5100 = 20150
        // VWAP = 20150 / 200 = 100.75
        assert!((bar.vwap - 100.75).abs() < 0.001);
    }

    #[test]
    fn test_tick_imbalance_bars() {
        let mut builder = BarBuilder::new(BarType::TickImbalance { threshold: 2 });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100);
        let tick2 = create_test_tick(base_time + Duration::seconds(1), 101.0, 100); // +1
        let tick3 = create_test_tick(base_time + Duration::seconds(2), 102.0, 100); // +1
        let tick4 = create_test_tick(base_time + Duration::seconds(3), 103.0, 100); // +1

        // Build up tick imbalance: 0, +1, +2, +3
        assert!(builder.process_tick(&tick1).is_none()); // imbalance = 0
        assert!(builder.process_tick(&tick2).is_none()); // imbalance = +1
        assert!(builder.process_tick(&tick3).is_none()); // imbalance = +2

        // Should complete when imbalance exceeds threshold (2)
        let bar = builder.process_tick(&tick4); // imbalance = +3 > 2
        assert!(bar.is_some());
    }

    #[test]
    fn test_run_bars() {
        let mut builder = BarBuilder::new(BarType::RunBars { run_length: 3 });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100);
        let tick2 = create_test_tick(base_time + Duration::seconds(1), 101.0, 100); // Up
        let tick3 = create_test_tick(base_time + Duration::seconds(2), 102.0, 100); // Up
        let tick4 = create_test_tick(base_time + Duration::seconds(3), 103.0, 100); // Up
        let tick5 = create_test_tick(base_time + Duration::seconds(4), 102.0, 100); // Down - direction change

        // Build up run of 3 up ticks
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());
        assert!(builder.process_tick(&tick3).is_none());
        assert!(builder.process_tick(&tick4).is_none());

        // Direction change should complete bar when run length >= threshold
        let bar = builder.process_tick(&tick5);
        assert!(bar.is_some());

        let bar = bar.unwrap();
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 102.0); // Close with the direction-changing tick
    }

    #[test]
    fn test_imbalance_tracker() {
        let mut tracker = ImbalanceTracker::new(100);

        // Test tick imbalance
        tracker.update(1, 100, 100.0);
        tracker.update(1, 100, 101.0);
        tracker.update(-1, 100, 100.5);

        assert_eq!(tracker.tick_imbalance, 1); // +1 +1 -1 = 1
        assert_eq!(tracker.volume_imbalance, 100.0); // +100 +100 -100 = 100

        // Test threshold checking
        assert!(tracker.tick_imbalance_exceeded(0));
        assert!(!tracker.tick_imbalance_exceeded(2));
    }

    #[test]
    fn test_bar_vwap_calculation() {
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 300 });

        let base_time = OffsetDateTime::now_utc();
        let tick1 = create_test_tick(base_time, 100.0, 100); // $10,000
        let tick2 = create_test_tick(base_time + Duration::seconds(1), 200.0, 100); // $20,000
        let tick3 = create_test_tick(base_time + Duration::seconds(2), 150.0, 100); // $15,000

        builder.process_tick(&tick1);
        builder.process_tick(&tick2);
        let bar = builder.process_tick(&tick3).unwrap();

        // VWAP should be (10000 + 20000 + 15000) / 300 = 150.0
        assert!((bar.vwap - 150.0).abs() < 0.001);
    }
}
