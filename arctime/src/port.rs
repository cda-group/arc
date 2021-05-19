use kompact::prelude::*;
use std::marker::PhantomData;

use crate::data::*;

/// A port for transferring data.
#[derive(Debug)]
pub struct DataPort<T: DataReqs>(PhantomData<T>);

impl<T: DataReqs> Port for DataPort<T> {
    type Indication = DataReply;
    type Request = DataEvent<T>;
}

/// An event which may arrive on a DataPort.
#[derive(Debug, Clone)]
pub enum DataEvent<T: DataReqs> {
    Watermark(DateTime),
    Item(DateTime, T),
    End,
}

#[derive(Debug, Clone)]
pub enum DataReply {
    Pull,
}

#[derive(Debug)]
pub struct CtrlPort;

impl Port for CtrlPort {
    type Indication = CtrlReply;
    type Request = CtrlEvent;
}

#[derive(Debug, Clone)]
pub enum CtrlEvent {}

#[derive(Debug, Clone)]
pub enum CtrlReply {}
