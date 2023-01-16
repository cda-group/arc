use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::marker::PhantomData;

use time::Duration;
use time::OffsetDateTime;

use crate::aggregator::Aggregator;
use crate::data::Data;
use crate::event::Event;
use crate::operator::Operator;

pub struct TumblingWindow<Iter, A: Aggregator> {
    input: Iter,
    param: Param<A>,
    state: State<A::P>,
}

pub struct State<P> {
    window: BTreeMap<i64, P>,
    aggregates: Option<(BTreeMap<i64, P>, i64)>,
}

pub struct Param<A> {
    length: i64,
    aggregator: A,
}

impl<Iter, A> TumblingWindow<Iter, A>
where
    A: Aggregator,
{
    fn new(
        input: Iter,
        every: Duration,
        length: Duration,
        aggregator: A,
    ) -> (Param<A>, State<A::P>) {
        (
            Param {
                length: length.whole_seconds(),
                aggregator,
            },
            State {
                window: BTreeMap::new(),
                aggregates: None,
            },
        )
    }
}

impl<Iter, A> TumblingWindow<Iter, A>
where
    A: Aggregator,
{
    fn process(input: Iter, param: Param<A>, state: State<A::P>) -> Self {
        Self {
            input,
            param,
            state,
        }
    }
}

impl<Iter, A> Iterator for TumblingWindow<Iter, A>
where
    Iter: Iterator<Item = Event<A::I>>,
    A: Aggregator,
{
    type Item = Event<A::O>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state.aggregates.as_mut() {
            None => loop {
                match self.input.next()? {
                    Event::Data(Some(time), data) => {
                        let slot = time.unix_timestamp() % self.param.length;
                        let agg = self.state.window.entry(slot).or_insert_with(A::identity);
                        *agg = A::merge(A::lift(data), agg.clone());
                        continue;
                    }
                    Event::Data(None, _) => break None,
                    Event::Watermark(time) => {
                        let time = time.unix_timestamp();
                        let slot = time % self.param.length;
                        let expired = self.state.window.split_off(&slot);
                        self.state.aggregates = Some((expired, time));
                    }
                }
            },
            Some((aggs, time)) => {
                if let Some((time, agg)) = aggs.pop_first() {
                    let time = Some(
                        OffsetDateTime::from_unix_timestamp(time * self.param.length).unwrap(),
                    );
                    return Some(Event::Data(time, (A::lower)(agg)));
                } else {
                    let time: OffsetDateTime = OffsetDateTime::from_unix_timestamp(*time).unwrap();
                    self.state.aggregates = None;
                    return Some(Event::Watermark(time));
                }
            }
        }
    }
}

impl<Iter, A> Operator for TumblingWindow<Iter, A>
where
    Iter: Iterator<Item = Event<A::I>>,
    A: Aggregator,
{
    type S = ();
    fn state(self) -> Self::S {
        ()
    }
}
