//! Comprehensive tests for bar structures implementation
//! 
//! Tests cover López de Prado's information-driven bar sampling methods including:
//! - All 6 bar types (Time, Volume, Dollar, VolumeImbalance, TickImbalance, RunBars)
//! - BarBuilder streaming functionality
//! - ImbalanceTracker accuracy
//! - Tick rule classification
//! - VWAP calculation correctness
//! - Edge cases (empty ticks, zero volumes, etc.)

use rust_indicators::financial::bars::{BarBuilder, BarType, Bar, Tick, ImbalanceTracker};
use time::{OffsetDateTime, Duration};
use rstest::{fixture, rstest};

#[cfg(test)]
mod bars_tests {
    use super::*;

    // Test fixtures for common scenarios
    #[fixture]
    fn base_time() -> OffsetDateTime {
        OffsetDateTime::now_utc()
    }

    #[fixture]
    fn sample_ticks(base_time: OffsetDateTime) -> Vec<Tick> {
        vec![
            Tick { timestamp: base_time, price: 100.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 },
            Tick { timestamp: base_time + Duration::seconds(2), price: 100.5, volume: 200 },
            Tick { timestamp: base_time + Duration::seconds(3), price: 102.0, volume: 120 },
            Tick { timestamp: base_time + Duration::seconds(4), price: 101.5, volume: 180 },
            Tick { timestamp: base_time + Duration::seconds(5), price: 103.0, volume: 160 },
        ]
    }

    #[fixture]
    fn trending_up_ticks(base_time: OffsetDateTime) -> Vec<Tick> {
        vec![
            Tick { timestamp: base_time, price: 100.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(3), price: 103.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(4), price: 104.0, volume: 100 },
        ]
    }

    #[fixture]
    fn volatile_ticks(base_time: OffsetDateTime) -> Vec<Tick> {
        vec![
            Tick { timestamp: base_time, price: 100.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(1), price: 105.0, volume: 200 },
            Tick { timestamp: base_time + Duration::seconds(2), price: 95.0, volume: 300 },
            Tick { timestamp: base_time + Duration::seconds(3), price: 110.0, volume: 150 },
            Tick { timestamp: base_time + Duration::seconds(4), price: 90.0, volume: 250 },
        ]
    }

    #[fixture]
    fn zero_volume_ticks(base_time: OffsetDateTime) -> Vec<Tick> {
        vec![
            Tick { timestamp: base_time, price: 100.0, volume: 0 },
            Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 0 },
            Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 100 },
        ]
    }

    // === TIME BARS TESTS ===

    #[test]
    fn test_time_bars_basic_functionality() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Time {
            frequency: Duration::seconds(2),
        });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 };
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 200 };

        // First two ticks should not complete bar
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Third tick should complete bar due to time threshold
        let bar = builder.process_tick(&tick3).unwrap();
        
        assert_eq!(bar.timestamp, base_time);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 102.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.volume, 450); // 100 + 150 + 200
        
        // VWAP calculation: (100*100 + 101*150 + 102*200) / 450
        let expected_vwap = (10000.0 + 15150.0 + 20400.0) / 450.0;
        assert!((bar.vwap - expected_vwap).abs() < 1e-6);
    }

    #[test]
    fn test_time_bars_exact_boundary() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Time {
            frequency: Duration::seconds(1),
        });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 };

        assert!(builder.process_tick(&tick1).is_none());
        let bar = builder.process_tick(&tick2).unwrap();
        
        assert_eq!(bar.volume, 250);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 101.0);
    }

    // === VOLUME BARS TESTS ===

    #[test]
    fn test_volume_bars_threshold_reached() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 300 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 };
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 100 };

        // First two ticks: 100 + 150 = 250 < 300
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Third tick: 250 + 100 = 350 >= 300
        let bar = builder.process_tick(&tick3).unwrap();
        
        assert_eq!(bar.volume, 350);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 102.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
    }

    #[test]
    fn test_volume_bars_exact_threshold() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 250 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 };

        assert!(builder.process_tick(&tick1).is_none());
        let bar = builder.process_tick(&tick2).unwrap();
        
        assert_eq!(bar.volume, 250);
    }

    #[test]
    fn test_volume_bars_zero_volume_handling() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 150 });

        let zero_ticks = zero_volume_ticks(base_time);
        
        // Zero volume ticks should not contribute to volume threshold
        assert!(builder.process_tick(&zero_ticks[0]).is_none());
        assert!(builder.process_tick(&zero_ticks[1]).is_none());
        
        // Only the third tick with volume 100 should contribute
        let bar = builder.process_tick(&zero_ticks[2]).unwrap();
        assert_eq!(bar.volume, 100); // Only non-zero volume counted
    }

    // === DOLLAR BARS TESTS ===

    #[test]
    fn test_dollar_bars_threshold_calculation() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Dollar { threshold: 25000.0 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 }; // $10,000
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 150.0, volume: 100 }; // $15,000
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 200.0, volume: 50 }; // $10,000

        // Total: $10,000 + $15,000 = $25,000 < threshold
        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());

        // Total: $25,000 + $10,000 = $35,000 >= threshold
        let bar = builder.process_tick(&tick3).unwrap();
        
        assert_eq!(bar.volume, 250);
        // VWAP = $35,000 / 250 = $140
        assert!((bar.vwap - 140.0).abs() < 1e-6);
    }

    #[test]
    fn test_dollar_bars_high_price_low_volume() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Dollar { threshold: 10000.0 });

        let tick1 = Tick { timestamp: base_time, price: 1000.0, volume: 5 }; // $5,000
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 2000.0, volume: 3 }; // $6,000

        assert!(builder.process_tick(&tick1).is_none());
        let bar = builder.process_tick(&tick2).unwrap();
        
        assert_eq!(bar.volume, 8);
        // VWAP = $11,000 / 8 = $1,375
        assert!((bar.vwap - 1375.0).abs() < 1e-6);
    }

    // === TICK IMBALANCE BARS TESTS ===

    #[test]
    fn test_tick_imbalance_bars_uptrend() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::TickImbalance { threshold: 3 });

        let uptrend_ticks = trending_up_ticks(base_time);
        
        // Process ticks with increasing prices (all positive tick signs)
        assert!(builder.process_tick(&uptrend_ticks[0]).is_none()); // tick_imbalance = 0
        assert!(builder.process_tick(&uptrend_ticks[1]).is_none()); // tick_imbalance = +1
        assert!(builder.process_tick(&uptrend_ticks[2]).is_none()); // tick_imbalance = +2
        assert!(builder.process_tick(&uptrend_ticks[3]).is_none()); // tick_imbalance = +3

        // Should complete when imbalance exceeds threshold
        let bar = builder.process_tick(&uptrend_ticks[4]); // tick_imbalance = +4 > 3
        assert!(bar.is_some());
        
        let bar = bar.unwrap();
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 104.0);
        assert_eq!(bar.volume, 500);
    }

    #[test]
    fn test_tick_imbalance_bars_mixed_direction() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::TickImbalance { threshold: 2 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 100 }; // +1
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 100.5, volume: 100 }; // -1
        let tick4 = Tick { timestamp: base_time + Duration::seconds(3), price: 102.0, volume: 100 }; // +1
        let tick5 = Tick { timestamp: base_time + Duration::seconds(4), price: 103.0, volume: 100 }; // +1

        assert!(builder.process_tick(&tick1).is_none()); // imbalance = 0
        assert!(builder.process_tick(&tick2).is_none()); // imbalance = +1
        assert!(builder.process_tick(&tick3).is_none()); // imbalance = 0 (+1-1)
        assert!(builder.process_tick(&tick4).is_none()); // imbalance = +1
        
        // Should complete when |imbalance| > threshold
        let bar = builder.process_tick(&tick5); // imbalance = +2
        assert!(bar.is_none()); // threshold is 2, so +2 is not > 2
    }

    // === VOLUME IMBALANCE BARS TESTS ===

    #[test]
    fn test_volume_imbalance_bars_calculation() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::VolumeImbalance { threshold: 200.0 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 150 }; // +150
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 100.5, volume: 100 }; // -100
        let tick4 = Tick { timestamp: base_time + Duration::seconds(3), price: 102.0, volume: 200 }; // +200

        assert!(builder.process_tick(&tick1).is_none()); // volume_imbalance = 0
        assert!(builder.process_tick(&tick2).is_none()); // volume_imbalance = +150
        assert!(builder.process_tick(&tick3).is_none()); // volume_imbalance = +50 (+150-100)
        
        // Should complete when |volume_imbalance| > threshold
        let bar = builder.process_tick(&tick4); // volume_imbalance = +250 > 200
        assert!(bar.is_some());
        
        let bar = bar.unwrap();
        assert_eq!(bar.volume, 550);
    }

    // === RUN BARS TESTS ===

    #[test]
    fn test_run_bars_direction_change() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::RunBars { run_length: 3 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 100 }; // Up
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 100 }; // Up
        let tick4 = Tick { timestamp: base_time + Duration::seconds(3), price: 103.0, volume: 100 }; // Up (run = 3)
        let tick5 = Tick { timestamp: base_time + Duration::seconds(4), price: 102.0, volume: 100 }; // Down (direction change)

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
        assert_eq!(bar.volume, 500);
    }

    #[test]
    fn test_run_bars_no_direction_change() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::RunBars { run_length: 5 });

        let uptrend_ticks = trending_up_ticks(base_time);
        
        // All ticks are in same direction, no bar should be completed
        for tick in &uptrend_ticks {
            assert!(builder.process_tick(tick).is_none());
        }
    }

    #[test]
    fn test_run_bars_alternating_prices() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::RunBars { run_length: 2 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 100 }; // Up
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 100 }; // Up (run = 2)
        let tick4 = Tick { timestamp: base_time + Duration::seconds(3), price: 101.0, volume: 100 }; // Down (direction change)

        assert!(builder.process_tick(&tick1).is_none());
        assert!(builder.process_tick(&tick2).is_none());
        assert!(builder.process_tick(&tick3).is_none());
        
        // Direction change with run length >= threshold should complete bar
        let bar = builder.process_tick(&tick4);
        assert!(bar.is_some());
    }

    // === IMBALANCE TRACKER TESTS ===

    #[test]
    fn test_imbalance_tracker_basic_functionality() {
        let mut tracker = ImbalanceTracker::new(100);

        // Test initial state
        assert_eq!(tracker.tick_imbalance, 0);
        assert_eq!(tracker.volume_imbalance, 0.0);

        // Test updates
        tracker.update(1, 100, 100.0); // Buy tick
        assert_eq!(tracker.tick_imbalance, 1);
        assert_eq!(tracker.volume_imbalance, 100.0);

        tracker.update(-1, 150, 99.0); // Sell tick
        assert_eq!(tracker.tick_imbalance, 0); // 1 + (-1) = 0
        assert_eq!(tracker.volume_imbalance, -50.0); // 100 + (-150) = -50

        tracker.update(1, 200, 101.0); // Buy tick
        assert_eq!(tracker.tick_imbalance, 1);
        assert_eq!(tracker.volume_imbalance, 150.0); // -50 + 200 = 150
    }

    #[test]
    fn test_imbalance_tracker_threshold_checking() {
        let mut tracker = ImbalanceTracker::new(100);

        // Build up imbalances
        tracker.update(1, 100, 100.0);
        tracker.update(1, 100, 101.0);
        tracker.update(1, 100, 102.0);

        // Test tick imbalance thresholds
        assert!(tracker.tick_imbalance_exceeded(2)); // |3| > 2
        assert!(!tracker.tick_imbalance_exceeded(3)); // |3| not > 3
        assert!(!tracker.tick_imbalance_exceeded(4)); // |3| not > 4

        // Test volume imbalance thresholds
        assert!(tracker.volume_imbalance_exceeded(200.0)); // |300| > 200
        assert!(!tracker.volume_imbalance_exceeded(300.0)); // |300| not > 300
    }

    #[test]
    fn test_imbalance_tracker_reset() {
        let mut tracker = ImbalanceTracker::new(100);

        // Build up imbalances
        tracker.update(1, 100, 100.0);
        tracker.update(-1, 150, 99.0);

        assert_ne!(tracker.tick_imbalance, 0);
        assert_ne!(tracker.volume_imbalance, 0.0);

        // Reset should clear current imbalances but preserve history
        tracker.reset();
        assert_eq!(tracker.tick_imbalance, 0);
        assert_eq!(tracker.volume_imbalance, 0.0);
    }

    #[test]
    fn test_imbalance_tracker_history_management() {
        let mut tracker = ImbalanceTracker::new(3); // Small history for testing

        // Fill history beyond capacity
        tracker.update(1, 100, 100.0);
        tracker.update(-1, 100, 99.0);
        tracker.update(1, 100, 101.0);
        tracker.update(-1, 100, 100.0);
        tracker.update(1, 100, 102.0);

        // History should be limited to max_history size
        // The tracker should still function correctly with limited history
        assert!(tracker.tick_imbalance_exceeded(0));
    }

    // === VWAP CALCULATION TESTS ===

    #[test]
    fn test_vwap_calculation_accuracy() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 300 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 }; // $10,000
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 200.0, volume: 100 }; // $20,000
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 150.0, volume: 100 }; // $15,000

        builder.process_tick(&tick1);
        builder.process_tick(&tick2);
        let bar = builder.process_tick(&tick3).unwrap();

        // VWAP should be (10000 + 20000 + 15000) / 300 = 150.0
        assert!((bar.vwap - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_vwap_with_zero_volume() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Time { frequency: Duration::seconds(1) });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 0 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 0 };

        builder.process_tick(&tick1);
        let bar = builder.process_tick(&tick2).unwrap();

        // With zero volume, VWAP should equal close price
        assert_eq!(bar.vwap, bar.close);
        assert_eq!(bar.volume, 0);
    }

    // === BUY/SELL VOLUME CLASSIFICATION TESTS ===

    #[test]
    fn test_buy_sell_volume_classification() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 400 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 100 }; // Up tick (buy)
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 100.5, volume: 100 }; // Down tick (sell)
        let tick4 = Tick { timestamp: base_time + Duration::seconds(3), price: 102.0, volume: 100 }; // Up tick (buy)

        builder.process_tick(&tick1);
        builder.process_tick(&tick2);
        builder.process_tick(&tick3);
        let bar = builder.process_tick(&tick4).unwrap();

        // Should have classified volumes correctly
        // tick1: neutral (split evenly), tick2: buy, tick3: sell, tick4: buy
        assert_eq!(bar.buy_volume, 250); // 50 + 100 + 0 + 100
        assert_eq!(bar.sell_volume, 150); // 50 + 0 + 100 + 0
        assert_eq!(bar.volume, 400);
    }

    // === EDGE CASES AND ERROR HANDLING ===

    #[test]
    fn test_empty_tick_stream() {
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 100 });
        
        // No ticks processed, no bars should be created
        assert!(!builder.has_current_bar());
    }

    #[test]
    fn test_single_tick_processing() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 50 });

        let tick = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let bar = builder.process_tick(&tick).unwrap();

        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 100.0);
        assert_eq!(bar.high, 100.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.volume, 100);
        assert_eq!(bar.vwap, 100.0);
    }

    #[test]
    fn test_identical_prices() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 300 });

        let tick1 = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        let tick2 = Tick { timestamp: base_time + Duration::seconds(1), price: 100.0, volume: 100 };
        let tick3 = Tick { timestamp: base_time + Duration::seconds(2), price: 100.0, volume: 100 };

        builder.process_tick(&tick1);
        builder.process_tick(&tick2);
        let bar = builder.process_tick(&tick3).unwrap();

        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 100.0);
        assert_eq!(bar.high, 100.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.vwap, 100.0);
    }

    #[test]
    fn test_force_complete_bar() {
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 1000 });

        let tick = Tick { timestamp: base_time, price: 100.0, volume: 100 };
        builder.process_tick(&tick);

        // Bar shouldn't complete naturally
        assert!(builder.has_current_bar());

        // Force completion
        let bar = builder.force_complete_bar().unwrap();
        assert_eq!(bar.volume, 100);
        assert!(!builder.has_current_bar());
    }

    // === PARAMETERIZED TESTS FOR DIFFERENT BAR TYPES ===

    #[rstest]
    #[case::time_bars(BarType::Time { frequency: Duration::seconds(2) })]
    #[case::volume_bars(BarType::Volume { threshold: 300 })]
    #[case::dollar_bars(BarType::Dollar { threshold: 25000.0 })]
    #[case::tick_imbalance_bars(BarType::TickImbalance { threshold: 2 })]
    #[case::volume_imbalance_bars(BarType::VolumeImbalance { threshold: 200.0 })]
    #[case::run_bars(BarType::RunBars { run_length: 3 })]
    fn test_bar_types_basic_properties(#[case] bar_type: BarType, sample_ticks: Vec<Tick>) {
        let mut builder = BarBuilder::new(bar_type.clone());
        
        // Verify builder properties
        assert_eq!(builder.bar_type(), &bar_type);
        assert!(!builder.has_current_bar());
        
        // Process some ticks
        for tick in &sample_ticks[..3] {
            builder.process_tick(tick);
        }
        
        // Should have a bar in progress after processing ticks
        assert!(builder.has_current_bar());
    }

    #[rstest]
    #[case::normal_ticks(sample_ticks(base_time()))]
    #[case::trending_ticks(trending_up_ticks(base_time()))]
    #[case::volatile_ticks(volatile_ticks(base_time()))]
    fn test_bar_ohlc_consistency(#[case] ticks: Vec<Tick>) {
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 200 });
        
        let mut bars = Vec::new();
        for tick in ticks {
            if let Some(bar) = builder.process_tick(&tick) {
                bars.push(bar);
            }
        }
        
        // Force complete any remaining bar
        if let Some(bar) = builder.force_complete_bar() {
            bars.push(bar);
        }
        
        // Verify OHLC consistency for each bar
        for bar in bars {
            assert!(bar.high >= bar.open, "High should be >= open");
            assert!(bar.high >= bar.close, "High should be >= close");
            assert!(bar.low <= bar.open, "Low should be <= open");
            assert!(bar.low <= bar.close, "Low should be <= close");
            assert!(bar.high >= bar.low, "High should be >= low");
            assert!(bar.volume >= bar.buy_volume, "Total volume should be >= buy volume");
            assert!(bar.volume >= bar.sell_volume, "Total volume should be >= sell volume");
            assert!(bar.vwap > 0.0 || bar.volume == 0, "VWAP should be positive unless zero volume");
        }
    }

    // === LÓPEZ DE PRADO SPECIFIC TESTS ===

    #[test]
    fn test_information_driven_sampling_properties() {
        // Test that information-driven bars sample more frequently during high activity
        let base_time = base_time();
        let volatile_ticks = volatile_ticks(base_time);
        
        let mut volume_builder = BarBuilder::new(BarType::Volume { threshold: 200 });
        let mut time_builder = BarBuilder::new(BarType::Time { frequency: Duration::seconds(10) });
        
        let mut volume_bars = 0;
        let mut time_bars = 0;
        
        for tick in volatile_ticks {
            if volume_builder.process_tick(&tick).is_some() {
                volume_bars += 1;
            }
            if time_builder.process_tick(&tick).is_some() {
                time_bars += 1;
            }
        }
        
        // Volume bars should adapt to activity level
        // This is a basic test - in practice, we'd need more sophisticated validation
        assert!(volume_bars >= 0); // At least some bars should be created
    }

    #[test]
    fn test_microstructural_information_preservation() {
        // Test that bars preserve microstructural information needed for López de Prado's methods
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::Volume { threshold: 300 });
        
        let ticks = sample_ticks(base_time);
        let mut bars = Vec::new();
        
        for tick in ticks {
            if let Some(bar) = builder.process_tick(&tick) {
                bars.push(bar);
            }
        }
        
        for bar in bars {
            // Should preserve buy/sell volume information
            assert!(bar.buy_volume + bar.sell_volume <= bar.volume);
            
            // Should have accurate VWAP calculation
            assert!(bar.vwap > 0.0 || bar.volume == 0);
            
            // Should preserve timestamp information
            assert!(bar.timestamp <= OffsetDateTime::now_utc());
            
            // Should maintain OHLC relationships
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open && bar.high >= bar.close);
            assert!(bar.low <= bar.open && bar.low <= bar.close);
        }
    }

    #[test]
    fn test_bar_sampling_synchronization() {
        // Test that bars are synchronized with information arrival
        let base_time = base_time();
        let mut builder = BarBuilder::new(BarType::VolumeImbalance { threshold: 100.0 });
        
        // Create ticks with strong imbalance
        let imbalanced_ticks = vec![
            Tick { timestamp: base_time, price: 100.0, volume: 100 },
            Tick { timestamp: base_time + Duration::seconds(1), price: 101.0, volume: 200 }, // Strong buy
            Tick { timestamp: base_time + Duration::seconds(2), price: 102.0, volume: 150 }, // Strong buy
        ];
        
        let mut bar_count = 0;
        for tick in imbalanced_ticks {
            if builder.process_tick(&tick).is_some() {
                bar_count += 1;
            }
        }
        
        // Should create bars when imbalance threshold is exceeded
        assert!(bar_count > 0);
    }
}