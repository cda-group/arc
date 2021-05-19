#![allow(deprecated)]

use hierarchical_hash_wheel_timer::wheels::quad_wheel::QuadWheelWithOverflow;
use hierarchical_hash_wheel_timer::wheels::Skip;
use hierarchical_hash_wheel_timer::wheels::TimerEntryWithDelay;
use hierarchical_hash_wheel_timer::TimerError;
use hierarchical_hash_wheel_timer::UuidOnlyTimerEntry as Entry;
use time::PrimitiveDateTime as DateTime;
use uuid::Uuid;

use crate::data::DataReqs;
use crate::task::Task;

use std::collections::HashMap;
use std::time::Duration;

pub struct EventTimer<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> {
    pub wheel: QuadWheelWithOverflow<Entry>,
    pub data: HashMap<Uuid, fn(&mut Task<S, I, O, R>)>,
}

type Callback<S, I, O, R> = fn(&mut Task<S, I, O, R>);

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Default for EventTimer<S, I, O, R> {
    fn default() -> Self {
        Self {
            wheel: QuadWheelWithOverflow::default(),
            data: HashMap::new(),
        }
    }
}

impl<S: DataReqs, I: DataReqs, O: DataReqs, R: DataReqs> Task<S, I, O, R> {
    /// Execute callback after duration
    pub fn after(&mut self, dur: Duration, cb: Callback<S, I, O, R>) {
        let entry = Entry::with_random_id(dur);
        self.etimer.data.insert(entry.id, cb);
        self.etimer.wheel.insert(entry).unwrap();
    }

    /// TODO: Handle overflow. Currently assumes Duration <= u32::MAX.
    pub(crate) fn advance(&mut self, mut remaining: Duration) {
        while remaining.as_millis() > 0 {
            match self.etimer.wheel.can_skip() {
                // No timers are scheduled
                Skip::Empty => {
                    self.time += remaining;
                }
                // Timers are scheduled at the next millisecond
                Skip::None => {
                    self.time += Duration::from_millis(1);
                    remaining -= Duration::from_millis(1);
                    for e in self.etimer.wheel.tick() {
                        (self.etimer.data.remove(&e.id).unwrap())(self);
                    }
                }
                // Timers are scheduled sometime later
                Skip::Millis(skip) => {
                    if skip as u128 >= remaining.as_millis() {
                        // No more entries to expire
                        self.etimer.wheel.skip(remaining.as_millis() as u32);
                        break;
                    } else {
                        // Skip until next entry
                        self.etimer.wheel.skip(skip);
                        self.time += Duration::from_millis(skip as u64);
                        remaining -= Duration::from_millis(skip as u64);
                    }
                }
            }
        }
    }
}
