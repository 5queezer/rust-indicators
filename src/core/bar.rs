use time::OffsetDateTime;

/// Represents a single OHLCV bar.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    pub timestamp: OffsetDateTime,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    /// Creates a new `Bar`.
    pub fn new(
        timestamp: OffsetDateTime,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

/// A generator for creating volume-based bars.
///
/// This struct accumulates trades and creates a new `Bar` whenever the
/// accumulated volume reaches a specified threshold.
#[derive(Debug, Clone)]
pub struct VolumeBar {
    pub threshold: f64,
    // Internal state for the current bar under construction.
    current_timestamp: Option<OffsetDateTime>,
    current_open: Option<f64>,
    current_high: f64,
    current_low: f64,
    current_volume: f64,
}

impl VolumeBar {
    /// Creates a new `VolumeBar` generator with a given volume threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            current_timestamp: None,
            current_open: None,
            current_high: 0.0, // Will be overwritten by the first trade
            current_low: 0.0,  // Will be overwritten by the first trade
            current_volume: 0.0,
        }
    }

    /// Adds a trade to the current bar.
    ///
    /// If the accumulated volume since the last bar exceeds the threshold,
    /// a new `Bar` is returned and the internal state is reset.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - The timestamp of the trade.
    /// * `price` - The price of the trade.
    /// * `volume` - The volume of the trade.
    ///
    /// # Returns
    ///
    /// An `Option<Bar>` which is `Some` if a new bar is created, and `None` otherwise.
    pub fn add_trade(
        &mut self,
        timestamp: OffsetDateTime,
        price: f64,
        volume: f64,
    ) -> Option<Bar> {
        if self.current_open.is_none() {
            // First trade for the new bar
            self.current_timestamp = Some(timestamp);
            self.current_open = Some(price);
            self.current_high = price;
            self.current_low = price;
        } else {
            self.current_high = self.current_high.max(price);
            self.current_low = self.current_low.min(price);
        }

        self.current_volume += volume;

        if self.current_volume >= self.threshold {
            let bar = Bar::new(
                self.current_timestamp.unwrap(),
                self.current_open.unwrap(),
                self.current_high,
                self.current_low,
                price, // Close price is the price of the last trade
                self.current_volume,
            );

            // Reset state for the next bar
            self.current_timestamp = None;
            self.current_open = None;
            self.current_volume = 0.0;

            return Some(bar);
        }

        None
    }
}

/// A generator for creating dollar-based bars.
///
/// This struct accumulates trades and creates a new `Bar` whenever the
/// accumulated dollar value of trades reaches a specified threshold.
#[derive(Debug, Clone)]
pub struct DollarBar {
    pub threshold: f64,
    // Internal state for the current bar under construction.
    current_timestamp: Option<OffsetDateTime>,
    current_open: Option<f64>,
    current_high: f64,
    current_low: f64,
    current_volume: f64,
    current_dollar_value: f64,
}

impl DollarBar {
    /// Creates a new `DollarBar` generator with a given dollar value threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            current_timestamp: None,
            current_open: None,
            current_high: 0.0,
            current_low: 0.0,
            current_volume: 0.0,
            current_dollar_value: 0.0,
        }
    }

    /// Adds a trade to the current bar.
    ///
    /// If the accumulated dollar value since the last bar exceeds the threshold,
    /// a new `Bar` is returned and the internal state is reset.
    pub fn add_trade(
        &mut self,
        timestamp: OffsetDateTime,
        price: f64,
        volume: f64,
    ) -> Option<Bar> {
        if self.current_open.is_none() {
            self.current_timestamp = Some(timestamp);
            self.current_open = Some(price);
            self.current_high = price;
            self.current_low = price;
        } else {
            self.current_high = self.current_high.max(price);
            self.current_low = self.current_low.min(price);
        }

        self.current_volume += volume;
        self.current_dollar_value += price * volume;

        if self.current_dollar_value >= self.threshold {
            let bar = Bar::new(
                self.current_timestamp.unwrap(),
                self.current_open.unwrap(),
                self.current_high,
                self.current_low,
                price, // Close price
                self.current_volume,
            );

            // Reset state
            self.current_timestamp = None;
            self.current_open = None;
            self.current_volume = 0.0;
            self.current_dollar_value = 0.0;

            return Some(bar);
        }

        None
    }
}

/// A generator for creating tick-based bars.
///
/// This struct accumulates trades and creates a new `Bar` whenever the
/// number of trades (ticks) reaches a specified threshold.
#[derive(Debug, Clone)]
pub struct TickBar {
    pub threshold: u32,
    // Internal state for the current bar under construction.
    current_timestamp: Option<OffsetDateTime>,
    current_open: Option<f64>,
    current_high: f64,
    current_low: f64,
    current_volume: f64,
    current_tick_count: u32,
}

impl TickBar {
    /// Creates a new `TickBar` generator with a given tick count threshold.
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            current_timestamp: None,
            current_open: None,
            current_high: 0.0,
            current_low: 0.0,
            current_volume: 0.0,
            current_tick_count: 0,
        }
    }

    /// Adds a trade to the current bar.
    ///
    /// If the number of ticks since the last bar exceeds the threshold,
    /// a new `Bar` is returned and the internal state is reset.
    pub fn add_trade(
        &mut self,
        timestamp: OffsetDateTime,
        price: f64,
        volume: f64,
    ) -> Option<Bar> {
        if self.current_open.is_none() {
            self.current_timestamp = Some(timestamp);
            self.current_open = Some(price);
            self.current_high = price;
            self.current_low = price;
        } else {
            self.current_high = self.current_high.max(price);
            self.current_low = self.current_low.min(price);
        }

        self.current_volume += volume;
        self.current_tick_count += 1;

        if self.current_tick_count >= self.threshold {
            let bar = Bar::new(
                self.current_timestamp.unwrap(),
                self.current_open.unwrap(),
                self.current_high,
                self.current_low,
                price, // Close price
                self.current_volume,
            );

            // Reset state
            self.current_timestamp = None;
            self.current_open = None;
            self.current_volume = 0.0;
            self.current_tick_count = 0;

            return Some(bar);
        }

        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    #[test]
    fn test_volume_bar_generator() {
        let mut volume_bar_generator = VolumeBar::new(100.0);

        // First trade, should not create a bar
        let result1 = volume_bar_generator.add_trade(datetime!(2023-01-01 12:00:00 UTC), 100.0, 50.0);
        assert!(result1.is_none());

        // Second trade, should not create a bar
        let result2 = volume_bar_generator.add_trade(datetime!(2023-01-01 12:00:01 UTC), 102.0, 40.0);
        assert!(result2.is_none());

        // Third trade, should create a bar
        let result3 = volume_bar_generator.add_trade(datetime!(2023-01-01 12:00:02 UTC), 101.0, 20.0);
        assert!(result3.is_some());

        let bar = result3.unwrap();
        assert_eq!(bar.timestamp, datetime!(2023-01-01 12:00:00 UTC));
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.close, 101.0);
        assert_eq!(bar.volume, 110.0);

        // Check if state is reset
        assert!(volume_bar_generator.current_open.is_none());
        assert_eq!(volume_bar_generator.current_volume, 0.0);

        // Add another trade to start a new bar
        let result4 = volume_bar_generator.add_trade(datetime!(2023-01-01 12:00:03 UTC), 105.0, 30.0);
        assert!(result4.is_none());
        assert_eq!(volume_bar_generator.current_open.unwrap(), 105.0);
        assert_eq!(volume_bar_generator.current_volume, 30.0);
    }

    #[test]
    fn test_dollar_bar_generator() {
        let mut dollar_bar_generator = DollarBar::new(10000.0);

        // First trade: 100 * 50 = 5000
        let result1 = dollar_bar_generator.add_trade(datetime!(2023-01-01 12:00:00 UTC), 100.0, 50.0);
        assert!(result1.is_none());

        // Second trade: 102 * 40 = 4080. Total dollar value = 5000 + 4080 = 9080
        let result2 = dollar_bar_generator.add_trade(datetime!(2023-01-01 12:00:01 UTC), 102.0, 40.0);
        assert!(result2.is_none());

        // Third trade: 101 * 20 = 2020. Total dollar value = 9080 + 2020 = 11100
        let result3 = dollar_bar_generator.add_trade(datetime!(2023-01-01 12:00:02 UTC), 101.0, 20.0);
        assert!(result3.is_some());

        let bar = result3.unwrap();
        assert_eq!(bar.timestamp, datetime!(2023-01-01 12:00:00 UTC));
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.close, 101.0);
        assert_eq!(bar.volume, 110.0); // 50 + 40 + 20

        // Check if state is reset
        assert!(dollar_bar_generator.current_open.is_none());
        assert_eq!(dollar_bar_generator.current_dollar_value, 0.0);
    }

    #[test]
    fn test_tick_bar_generator() {
        let mut tick_bar_generator = TickBar::new(3);

        // First trade
        let result1 = tick_bar_generator.add_trade(datetime!(2023-01-01 12:00:00 UTC), 100.0, 50.0);
        assert!(result1.is_none());

        // Second trade
        let result2 = tick_bar_generator.add_trade(datetime!(2023-01-01 12:00:01 UTC), 102.0, 40.0);
        assert!(result2.is_none());

        // Third trade, should create a bar
        let result3 = tick_bar_generator.add_trade(datetime!(2023-01-01 12:00:02 UTC), 101.0, 20.0);
        assert!(result3.is_some());

        let bar = result3.unwrap();
        assert_eq!(bar.timestamp, datetime!(2023-01-01 12:00:00 UTC));
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.close, 101.0);
        assert_eq!(bar.volume, 110.0);

        // Check if state is reset
        assert!(tick_bar_generator.current_open.is_none());
        assert_eq!(tick_bar_generator.current_tick_count, 0);
    }
}
