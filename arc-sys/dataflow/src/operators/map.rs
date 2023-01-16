use serde::Deserialize;
use serde::Serialize;

use crate::data::Data;
use crate::operator::Operator;

use crate::event::Event;

pub struct Map<Iter, I, O> {
    input: Iter,
    param: Param<I, O>,
    state: State,
}

impl<I, O> Map<(), I, O> {
    pub const fn new(fun: fn(I) -> O) -> (Param<I, O>, State) {
        (Param { fun }, State {})
    }
}

impl<Iter, I, O> Map<Iter, I, O> {
    pub const fn process(input: Iter, params: Param<I, O>, state: State) -> Self {
        Self {
            input,
            param: params,
            state,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State;

#[derive(Copy, Clone)]
pub struct Param<I, O> {
    fun: fn(I) -> O,
}

impl<Iter, I, O> Operator for Map<Iter, I, O>
where
    Iter: Iterator<Item = Event<I>>,
    I: Data,
    O: Data,
{
    type S = State;
    fn state(self) -> Self::S {
        State
    }
}

impl<Iter, I, O> Iterator for Map<Iter, I, O>
where
    Iter: Iterator<Item = Event<I>>,
    I: Data,
    O: Data,
{
    type Item = Event<O>;
    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|e| e.map(self.param.fun))
    }
}

impl State {
    pub const fn new<I, O>(fun: fn(I) -> O) -> (Param<I, O>, State) {
        (Param { fun }, State {})
    }
}
