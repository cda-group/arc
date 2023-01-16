use std::cell::UnsafeCell;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::Serialize;
use time::OffsetDateTime;

use crate::event::Event;
use crate::iteratee::Iteratee;
use crate::operator::Operator;
use crate::prelude::Data;

use super::transform::TransformIteratee;

pub struct Apply<Iter0, Iter1, I, O> {
    iter0: Iter0,
    iter1: Iter1,
    time: Option<OffsetDateTime>,
    param: Param,
    state: State,
    _i: PhantomData<I>,
    _o: PhantomData<O>,
}

impl Apply<(), (), (), ()> {
    pub const fn new() -> (Param, State) {
        (Param, State)
    }
}

impl<Iter0, Iter1, I, O> Apply<Iter0, Iter1, I, O> {
    pub const fn process(
        iter0: Iter0,
        iter1: Iter1,
        param: Param,
        state: State,
    ) -> Apply<Iter0, Iter1, I, O> {
        Self {
            iter0,
            iter1,
            time: None,
            param,
            state,
            _i: PhantomData,
            _o: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State;

#[derive(Clone, Copy)]
pub struct Param;

impl<Iter0, Iter1, I: Data, O: Data> Iterator for Apply<Iter0, Iter1, I, O>
where
    Iter0: Iterator<Item = Event<I>>,
    Iter1: Iteratee<Item = I> + Iterator<Item = O>,
{
    type Item = Event<O>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter1.next() {
                Some(data) => return Some(Event::Data(self.time, data)),
                None => match self.iter0.next() {
                    Some(Event::Data(time, data)) => {
                        self.iter1.feed(data);
                        self.time = time;
                        continue;
                    }
                    Some(Event::Watermark(t)) => return Some(Event::Watermark(t)),
                    None => return None,
                },
            }
        }
    }
}

impl<Iter0, Iter1, I: Data, O: Data> Operator for Apply<Iter0, Iter1, I, O>
where
    Iter0: Iterator<Item = Event<I>>,
    Iter1: Iteratee<Item = I> + Iterator<Item = O>,
{
    type S = State;
    fn state(self) -> Self::S {
        self.state
    }
}
