//! Builtin streaming operators. Everything required to support the SQL-interface.
#![allow(clippy::type_complexity)]

use crate::data::Sharable;
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

impl<I: Sharable> clm::Pullable<I> {
    pub fn iterate<O: Sharable>(self, f: fn(Self) -> (Self, clm::Pullable<O>)) -> clm::Pullable<O> {
        todo!()
    }

//     pub fn key_by<K: Sharable>(self, f: fn(I) -> K) -> cld::Pullable<I> {
//         todo!()
//     }

    pub fn map<O: Sharable>(self, f: fn(I) -> O) -> clm::Pullable<O> {
        todo!()
    }

    pub fn filter(self, f: fn(I) -> bool) -> clm::Pullable<I> {
        todo!()
    }

    pub fn flat_map<O: Sharable>(self, f: fn(I) -> clm::Pullable<O>) -> clm::Pullable<O> {
        todo!()
    }

    pub fn reduce<O: Sharable>(self, f: fn(O, I) -> O) -> clm::Pullable<O> {
        todo!()
    }

//     pub fn join<K: Sharable + Hash>(
//         self,
//         other: clw::Pullable<I>,
//         f: fn(I, I) -> K,
//     ) -> clm::Pullable<I> {
//         todo!()
//     }
//
//     pub fn tumbling_window<O: Sharable>(self, len: Duration) -> clw::Pullable<O> {
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

impl<T: Sharable> Iterator for DataGen<T>
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
