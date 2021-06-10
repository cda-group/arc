#![allow(dead_code)]

use arctime::prelude::*;
use std::marker::PhantomData;

struct DataGen<T> {
    count: usize,
    marker: PhantomData<T>,
}

impl<T: DataReqs> Iterator for DataGen<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
