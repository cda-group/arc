//! Everything related to ports and what can be sent over ports can be found in this module.
use kompact::prelude::*;
pub use time::PrimitiveDateTime as DateTime;

use std::fmt::Debug;
use std::marker::PhantomData;

/// Trait requirements for sending data on a port.
pub trait DataReqs: 'static + Send + Debug + Clone {}
impl<T> DataReqs for T where T: 'static + Send + Debug + Clone {}

/// A datatype which can be used for sending values of either type `L` or `R`.
#[derive(Debug, Clone)]
pub enum Either<L, R> {
    L(L),
    R(R),
}

/// A port for transferring data.
#[derive(Debug)]
pub struct StreamPort<T: DataReqs>(PhantomData<T>);

impl<T: DataReqs> Port for StreamPort<T> {
    type Indication = StreamReply;
    type Request = StreamEvent<T>;
}

/// An message sent from producers to consumers which may arrive on a `DataPort`.
#[derive(Debug, Clone)]
pub enum StreamEvent<T: DataReqs> {
    Watermark(DateTime),
    Data(DateTime, T),
    End,
}

/// A message sent from consumers to producers which may arrive on a `ControlPort`.
#[derive(Debug, Clone)]
pub enum StreamReply {
    Pull,
}

/// A port for transferring ad-hoc control information.
#[derive(Debug)]
pub struct ControlPort;

impl Port for ControlPort {
    type Indication = ControlReply;
    type Request = ControlEvent;
}

/// A message sent from a task manager to a task.
#[derive(Debug, Clone)]
pub enum ControlEvent {}

/// A message sent from a task to a task manager.
#[derive(Debug, Clone)]
pub enum ControlReply {}
