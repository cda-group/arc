use serde::Deserialize;
use serde::Serialize;
use time::OffsetDateTime;

use crate::data::Data;
use crate::event::Event;
use crate::operator::Operator;

pub struct Union<Iter0, Iter1> {
    input0: Iter0,
    input1: Iter1,
    params: Param,
    state: State,
}

impl Union<(), ()> {
    pub const fn new() -> (Param, State) {
        (Param, State { watermark: None })
    }
}

impl<Iter0, Iter1> Union<Iter0, Iter1> {
    pub const fn process(input0: Iter0, input1: Iter1, params: Param, state: State) -> Self {
        Self {
            input0,
            input1,
            params,
            state,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State {
    watermark: Option<Latest>,
}

#[derive(Clone, Copy)]
pub struct Param;

// The latest watermark.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Latest {
    Iter0(OffsetDateTime),
    Iter1(OffsetDateTime),
}

impl<Iter0, Iter1, I> Iterator for Union<Iter0, Iter1>
where
    I: Data,
    Iter0: Iterator<Item = Event<I>>,
    Iter1: Iterator<Item = Event<I>>,
{
    type Item = Event<I>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.input0.next() {
                Some(Event::Watermark(ta)) => match self.state.watermark {
                    Some(Latest::Iter1(tb)) => {
                        if ta > tb {
                            self.state.watermark = Some(Latest::Iter0(ta));
                            return Some(Event::Watermark(tb));
                        }
                    }
                    _ => self.state.watermark = Some(Latest::Iter0(ta)),
                },
                Some(Event::Data(t0, data)) => return Some(Event::Data(t0, data)),
                None => {}
            }
            match self.input1.next() {
                Some(Event::Watermark(tb)) => match self.state.watermark {
                    Some(Latest::Iter0(ta)) => {
                        if tb > ta {
                            self.state.watermark = Some(Latest::Iter1(tb));
                            return Some(Event::Watermark(ta));
                        }
                    }
                    _ => self.state.watermark = Some(Latest::Iter1(tb)),
                },
                Some(Event::Data(t1, data)) => return Some(Event::Data(t1, data)),
                None => return None,
            }
        }
    }
}

impl<Iter0, Iter1, I> Operator for Union<Iter0, Iter1>
where
    I: Data,
    Iter0: Iterator<Item = Event<I>>,
    Iter1: Iterator<Item = Event<I>>,
{
    type S = State;
    fn state(self) -> Self::S {
        self.state
    }
}
