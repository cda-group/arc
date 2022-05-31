//! Builtin streaming operators. Everything required to support the SQL-interface.
#![allow(clippy::type_complexity)]

use crate::data::Data;
use crate::prelude::*;

use crate::data::channels::local::multicast as clm;
// use crate::channels::local::data_parallel as cld;
// use crate::channels::local::data_parallel as clt;
// use crate::channels::local::window as clw;

use crate::prelude::DateTime;

use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

use std::marker::PhantomData;

use kompact::prelude::*;

use std::sync::Arc;

impl<I: Data> clm::PullChan<I> {
    pub fn iterate<O: Data>(self, f: fn(Self) -> (Self, clm::PullChan<O>)) -> clm::PullChan<O> {
        todo!()
    }

    //     pub fn key_by<K: Data>(self, f: fn(I) -> K) -> cld::PullChan<I> {
    //         todo!()
    //     }

    pub fn map<O: Data>(self, f: fn(I) -> O) -> clm::PullChan<O> {
        todo!()
    }

    pub fn filter(self, f: fn(I) -> bool) -> clm::PullChan<I> {
        todo!()
    }

    pub fn flat_map<O: Data>(self, f: fn(I) -> clm::PullChan<O>) -> clm::PullChan<O> {
        todo!()
    }

    pub fn reduce<O: Data>(self, f: fn(O, I) -> O) -> clm::PullChan<O> {
        todo!()
    }

    //     pub fn join<K: Data + Hash>(
    //         self,
    //         other: clw::PullChan<I>,
    //         f: fn(I, I) -> K,
    //     ) -> clm::PullChan<I> {
    //         todo!()
    //     }
    //
    //     pub fn tumbling_window<O: Data>(self, len: Duration) -> clw::PullChan<O> {
    //         todo!()
    //     }
}

pub struct DataGen<T> {
    offset: i64,
    count: usize,
    rng: rand::prelude::ThreadRng,
    marker: PhantomData<T>,
}

impl<T> DataGen<T> {
    fn new(count: usize) -> Self {
        Self {
            offset: 0,
            count,
            rng: rand::prelude::thread_rng(),
            marker: PhantomData,
        }
    }
}

impl<T: Data> Iterator for DataGen<T>
where
    Standard: Distribution<T>,
{
    type Item = (DateTime, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.offset += 1;
        self.count -= 1;
        if self.count > 0 {
            todo!()
        //             Some((DateTime::from_unix_timestamp(self.offset), self.rng.gen()))
        } else {
            None
        }
    }
}
