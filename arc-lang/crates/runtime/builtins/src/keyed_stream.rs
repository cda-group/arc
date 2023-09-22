#![allow(irrefutable_let_patterns)]
use std::collections::BTreeMap;
use std::collections::HashMap;

use num::Integer;
use time::OffsetDateTime;
use tokio::sync::mpsc::Receiver;

use crate::aggregator::Aggregator;
use crate::dict::Dict;
use crate::duration::Duration;
use crate::stream::Event;
use crate::stream::Stream;
use crate::time::Time;
use crate::traits::Data;
use crate::traits::Key;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum KeyedEvent<K, T> {
    Data(Time, K, T),
    Watermark(Time),
    Snapshot(usize),
    Sentinel,
}

pub struct KeyedStream<K: Data, T: Data>(pub(crate) Receiver<KeyedEvent<K, T>>);

impl<K: Key, T: Data> KeyedStream<K, T> {
    pub fn map<O>(mut self, f: fn(T) -> O) -> KeyedStream<K, O>
    where
        O: Data,
    {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(t, k, v) => tx1.send(KeyedEvent::Data(t, k, f(v))).await,
                    KeyedEvent::Watermark(t) => tx1.send(KeyedEvent::Watermark(t)).await,
                    KeyedEvent::Snapshot(i) => tx1.send(KeyedEvent::Snapshot(i)).await,
                    KeyedEvent::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
                .unwrap()
            }
        });
        KeyedStream(rx1)
    }
    pub fn filter(mut self, f: fn(T) -> bool) -> KeyedStream<K, T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(t, k, v) => {
                        if f(v.clone()) {
                            tx1.send(KeyedEvent::Data(t, k, v)).await.unwrap();
                        }
                    }
                    KeyedEvent::Watermark(t) => tx1.send(KeyedEvent::Watermark(t)).await.unwrap(),
                    KeyedEvent::Snapshot(i) => tx1.send(KeyedEvent::Snapshot(i)).await.unwrap(),
                    KeyedEvent::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        KeyedStream(rx1)
    }

    pub fn join<T1, T2>(mut self, index: Dict<K, T1>, merge: fn(T, T1) -> T2) -> KeyedStream<K, T2>
    where
        T1: Data,
        T2: Data,
    {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(t, k, v0) => {
                        if let Some(v1) = index.0.get(&k) {
                            let v2 = merge(v0, v1.clone());
                            tx1.send(KeyedEvent::Data(t, k, v2)).await.unwrap();
                        }
                    }
                    KeyedEvent::Watermark(t) => {
                        tx1.send(KeyedEvent::Watermark(t)).await.unwrap();
                    }
                    KeyedEvent::Snapshot(i) => {
                        tx1.send(KeyedEvent::Snapshot(i)).await.unwrap();
                    }
                    KeyedEvent::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        KeyedStream(rx1)
    }

    pub fn merge(mut self, mut other: Self) -> Self {
        let (tx2, rx2) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                let event = tokio::select! {
                    event = self.recv() => {
                        if let KeyedEvent::Sentinel = event {
                            other.recv().await
                        } else {
                            event
                        }
                    },
                    event = other.recv() => {
                        if let KeyedEvent::Sentinel = event {
                            self.recv().await
                        } else {
                            event
                        }
                    },
                };
                match event {
                    KeyedEvent::Data(t, k1, v1) => {
                        tx2.send(KeyedEvent::Data(t, k1, v1)).await.unwrap();
                    }
                    KeyedEvent::Watermark(t) => {
                        tx2.send(KeyedEvent::Watermark(t)).await.unwrap();
                    }
                    KeyedEvent::Snapshot(i) => {
                        tx2.send(KeyedEvent::Snapshot(i)).await.unwrap();
                    }
                    KeyedEvent::Sentinel => {
                        tx2.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        Self(rx2)
    }

    pub fn split(mut self) -> (Self, Self) {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        let (tx2, rx2) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                let (l, r) = match self.recv().await {
                    KeyedEvent::Data(t, k1, v1) => {
                        let k2 = k1.clone();
                        let v2 = v1.clone();
                        tokio::join!(
                            tx1.send(KeyedEvent::Data(t, k2, v2)),
                            tx2.send(KeyedEvent::Data(t, k1, v1)),
                        )
                    }
                    KeyedEvent::Watermark(t) => {
                        tokio::join!(
                            tx1.send(KeyedEvent::Watermark(t)),
                            tx2.send(KeyedEvent::Watermark(t))
                        )
                    }
                    KeyedEvent::Snapshot(i) => {
                        tokio::join!(
                            tx1.send(KeyedEvent::Snapshot(i)),
                            tx2.send(KeyedEvent::Snapshot(i))
                        )
                    }
                    KeyedEvent::Sentinel => {
                        tokio::join!(
                            tx1.send(KeyedEvent::Sentinel),
                            tx2.send(KeyedEvent::Sentinel)
                        )
                    }
                };
                l.unwrap();
                r.unwrap();
            }
        });
        (Self(rx1), Self(rx2))
    }

    pub fn scan<P, O>(
        mut self,
        agg: Aggregator<fn(T) -> P, fn(P, P) -> P, fn() -> P, fn(P) -> O>,
    ) -> KeyedStream<K, O>
    where
        P: Data,
        O: Data,
    {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        let mut state = HashMap::new();
        let Aggregator::Monoid {
            lift,
            combine,
            identity,
            lower,
        } = agg
        else {
            unreachable!()
        };
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(t, k, v) => {
                        let p = state.entry(k.clone()).or_insert_with(identity);
                        *p = combine(p.clone(), lift(v));
                        tx1.send(KeyedEvent::Data(t, k, lower(p.clone())))
                            .await
                            .unwrap();
                    }
                    KeyedEvent::Watermark(t) => {
                        tx1.send(KeyedEvent::Watermark(t)).await.unwrap();
                    }
                    KeyedEvent::Snapshot(i) => {
                        tx1.send(KeyedEvent::Snapshot(i)).await.unwrap();
                    }
                    KeyedEvent::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        KeyedStream(rx1)
    }

    #[allow(unstable_name_collisions)]
    pub fn window<P, O>(
        mut self,
        size: Duration,
        agg: Aggregator<fn(T) -> P, fn(P, P) -> P, fn() -> P, fn(P) -> O>,
    ) -> KeyedStream<K, O>
    where
        P: Data,
        O: Data,
    {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        let mut aggs: BTreeMap<OffsetDateTime, HashMap<K, P>> = BTreeMap::new();
        let Aggregator::Monoid {
            lift,
            combine,
            identity,
            lower,
        } = agg
        else {
            unreachable!()
        };
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(time, key, data) => {
                        let slot = time
                            .0
                            .unix_timestamp_nanos()
                            .div_floor(&size.0.whole_nanoseconds());
                        let time = OffsetDateTime::from_unix_timestamp_nanos(
                            slot * size.0.whole_nanoseconds(),
                        )
                        .expect("Error converting timestamp to OffsetDateTime.");
                        let p = aggs
                            .entry(time)
                            .or_insert_with(HashMap::new)
                            .entry(key)
                            .or_insert_with(identity);
                        *p = combine(p.clone(), lift(data));
                    }
                    KeyedEvent::Watermark(watermark) => {
                        while let Some(entry) = aggs.first_entry() {
                            let time = *entry.key() + size.0;
                            if time < watermark.0 {
                                for (key, p) in entry.remove() {
                                    tx1.send(KeyedEvent::Data(Time(time), key, lower(p)))
                                        .await
                                        .unwrap();
                                }
                                tx1.send(KeyedEvent::Watermark(watermark)).await.unwrap();
                            } else {
                                tx1.send(KeyedEvent::Watermark(watermark)).await.unwrap();
                                break;
                            }
                        }
                    }
                    KeyedEvent::Snapshot(i) => {
                        tx1.send(KeyedEvent::Snapshot(i)).await.unwrap();
                    }
                    KeyedEvent::Sentinel => {
                        tx1.send(KeyedEvent::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        KeyedStream(rx1)
    }

    pub fn unkey(mut self) -> Stream<T> {
        let (tx1, rx1) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_local(async move {
            loop {
                match self.recv().await {
                    KeyedEvent::Data(t, _, v) => {
                        tx1.send(Event::Data(t, v)).await.unwrap();
                    }
                    KeyedEvent::Watermark(t) => {
                        tx1.send(Event::Watermark(t)).await.unwrap();
                    }
                    KeyedEvent::Snapshot(i) => {
                        tx1.send(Event::Snapshot(i)).await.unwrap();
                    }
                    KeyedEvent::Sentinel => {
                        tx1.send(Event::Sentinel).await.unwrap();
                        break;
                    }
                }
            }
        });
        Stream(rx1)
    }

    async fn recv(&mut self) -> KeyedEvent<K, T> {
        self.0.recv().await.unwrap()
    }
}
