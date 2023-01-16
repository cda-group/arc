use serde::Deserialize;
use serde::Serialize;

use crate::data::Data;
use crate::event::Event;
use crate::operator::Operator;

pub struct Filter<Iter, T> {
    input: Iter,
    param: Param<T>,
    state: State,
}

impl<T> Filter<(), T> {
    pub const fn new(fun: fn(T) -> bool) -> (Param<T>, State) {
        (Param { fun }, State {})
    }
}

impl<Iter, T> Filter<Iter, T> {
    pub const fn process(input: Iter, param: Param<T>, state: State) -> Filter<Iter, T> {
        Self {
            input,
            param,
            state,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Param<T> {
    fun: fn(T) -> bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State;

impl<Iter, T> Operator for Filter<Iter, T>
where
    Iter: Iterator<Item = Event<T>>,
    T: Data,
{
    type S = State;
    fn state(self) -> Self::S {
        State
    }
}

impl<Iter, T> Iterator for Filter<Iter, T>
where
    Iter: Iterator<Item = Event<T>>,
    T: Data,
{
    type Item = Event<T>;
    fn next(&mut self) -> Option<Self::Item> {
        for item in &mut self.input {
            match item {
                Event::Data(t, data) => {
                    if (self.param.fun)(data.clone()) {
                        return Some(Event::Data(t, data));
                    }
                }
                Event::Watermark(t) => return Some(Event::Watermark(t)),
            }
        }
        None
    }
}
