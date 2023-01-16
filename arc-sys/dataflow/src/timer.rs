#![allow(deprecated)]

use uuid::Uuid;
use wheel::wheels::quad_wheel::QuadWheelWithOverflow;
use wheel::wheels::Skip;
use wheel::UuidOnlyTimerEntry as Entry;

use std::collections::HashMap;
use std::time::Duration;

/// A timer
pub struct Timer {
    pub wheel: QuadWheelWithOverflow<Entry>,
    pub callbacks: HashMap<Uuid, Callback>,
    pub duration: Duration,
}

type Callback = fn();

impl Default for Timer {
    fn default() -> Self {
        Self {
            wheel: QuadWheelWithOverflow::default(),
            callbacks: HashMap::new(),
            duration: Duration::from_millis(0),
        }
    }
}

impl Timer {
    /// Execute callback after duration
    pub fn after(&mut self, duration: Duration, callback: Callback) {
        let entry = Entry::with_random_id(duration);
        self.callbacks.insert(entry.id, callback);
        self.wheel.insert(entry).unwrap();
    }

    /// Advance the timer and execute timers which have expired.
    /// TODO: Handle overflow. Currently assumes Duration <= u32::MAX.
    pub(crate) fn advance(&mut self, mut remaining: Duration) {
        while remaining.as_millis() > 0 {
            match self.wheel.can_skip() {
                // No timers are scheduled
                Skip::Empty => {
                    self.duration += remaining;
                }
                // Timers are scheduled at the next millisecond
                Skip::None => {
                    self.duration += Duration::from_millis(1);
                    remaining -= Duration::from_millis(1);
                    for e in self.wheel.tick() {
                        (self.callbacks.remove(&e.id).unwrap())();
                    }
                }
                // Timers are scheduled sometime later
                Skip::Millis(skip) => {
                    if skip as u128 >= remaining.as_millis() {
                        // No more entries to expire
                        self.wheel.skip(remaining.as_millis() as u32);
                        break;
                    } else {
                        // Skip until next entry
                        self.wheel.skip(skip);
                        self.duration += Duration::from_millis(skip as u64);
                        remaining -= Duration::from_millis(skip as u64);
                    }
                }
            }
        }
    }
}
