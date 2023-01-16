use serde::Deserialize;
use serde::Serialize;

use crate::data::Data;
use crate::event::Event;
use crate::operator::Operator;

pub struct Scan<Iter, I, O> {
    input: Iter,
    params: Param<I, O>,
    state: State<O>,
}

impl<I, O> Scan<(), I, O> {
    pub const fn new(fun: fn(I, O) -> O, agg: O) -> (Param<I, O>, State<O>) {
        (Param { fun }, State { agg })
    }
}

impl<Iter, I, O> Scan<Iter, I, O> {
    pub const fn process(input: Iter, params: Param<I, O>, state: State<O>) -> Self {
        Self {
            input,
            params,
            state,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State<O> {
    agg: O,
}

#[derive(Copy, Clone)]
pub struct Param<I, O> {
    fun: fn(I, O) -> O,
}

impl<Iter, T, U> Iterator for Scan<Iter, T, U>
where
    Iter: Iterator<Item = Event<T>>,
    T: Data,
    U: Data,
{
    type Item = Event<U>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.input.next() {
            Some(Event::Data(time, data)) => {
                self.state.agg = (self.params.fun)(data, self.state.agg.clone());
                Some(Event::Data(time, self.state.agg.clone()))
            }
            Some(Event::Watermark(time)) => Some(Event::Watermark(time)),
            None => None,
        }
    }
}

impl<Iter, I, O> Operator for Scan<Iter, I, O>
where
    Iter: Iterator<Item = Event<I>>,
    I: Data,
    O: Data,
{
    type S = State<O>;
    fn state(self) -> Self::S {
        self.state
    }
}
