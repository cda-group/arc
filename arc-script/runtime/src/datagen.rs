#![allow(deprecated)]
#![allow(dead_code)]

use crate::port::DataReqs;
use crate::port::DateTime;

use rand::Rng;
use rand::distributions::Standard;
use rand::distributions::Distribution;

use std::marker::PhantomData;

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

impl<T: DataReqs> Iterator for DataGen<T>
where
    Standard: Distribution<T>,
{
    type Item = (DateTime, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.offset += 1;
        self.count -= 1;
        if self.count > 0 {
            Some((DateTime::from_unix_timestamp(self.offset), self.rng.gen::<T>()))
        } else {
            None
        }
    }
}
