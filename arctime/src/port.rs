use kompact::prelude::*;
use std::marker::PhantomData;

use std::fmt::Debug;
pub use time::PrimitiveDateTime as DateTime;

pub trait DataReqs: 'static + Sync + Send + Debug + Clone {}
impl<T> DataReqs for T where T: 'static + Sync + Send + Debug + Clone {}

#[derive(Debug, Clone)]
pub enum Either<L, R> {
    L(L),
    R(R),
}

/// A port which streams can connect to.
#[derive(Debug)]
pub struct StreamPort<T: DataReqs>(PhantomData<T>);

impl<T: DataReqs> Port for StreamPort<T> {
    type Indication = StreamReply;
    type Request = StreamEvent<T>;
}

/// An event which may arrive on a DataPort.
#[derive(Debug, Clone)]
pub enum StreamEvent<T: DataReqs> {
    Watermark(DateTime),
    Data(DateTime, T),
    End,
}

#[derive(Debug, Clone)]
pub enum StreamReply {
    Pull,
}

#[derive(Debug)]
pub struct ControlPort;

impl Port for ControlPort {
    type Indication = ControlReply;
    type Request = ControlEvent;
}

#[derive(Debug, Clone)]
pub enum ControlEvent {}

#[derive(Debug, Clone)]
pub enum ControlReply {}
