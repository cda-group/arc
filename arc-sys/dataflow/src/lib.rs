#![allow(non_snake_case)]
#![allow(unused)]
#![feature(type_alias_impl_trait)]
#![feature(poll_ready)]
#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(cell_update)]

pub mod aggregator;
pub mod data;
pub mod db;
pub mod event;
pub mod iteratee;
pub mod operator;
pub mod runtime;
pub mod serde;

#[cfg(feature = "ml")]
pub mod ml {
    pub mod image;
    pub mod model;
}

pub mod state;

pub mod operators {
    pub mod apply;
    pub mod compiler_generated;
    pub mod filter;
    pub mod kafka_sink;
    pub mod kafka_source;
    pub mod map;
    pub mod scan;
    pub mod shuffle_sink;
    pub mod shuffle_source;
    pub mod transform;
    pub mod tumbling_window;
    pub mod union;
    pub mod worker_sink;
    pub mod worker_source;
}

pub mod aggregators {
    pub mod count;
    pub mod sum;
}

pub mod utils {
    pub mod serde;
}

pub mod prelude {

    // Procedural macros
    pub use macros::access;
    pub use macros::call;
    pub use macros::call_async;
    pub use macros::call_indirect;
    pub use macros::enwrap;
    pub use macros::is;
    pub use macros::new;
    pub use macros::rewrite;
    pub use macros::unwrap;
    pub use macros::val;
    pub use macros::vector;
    pub use macros::NoDebug;
    pub use macros::NoSerde;
    pub use macros::Unpin;

    pub use derive_more::Deref;
    pub use derive_more::DerefMut;
    pub use derive_more::From;

    pub use crate::data::cell::Cell;
    pub use crate::data::cell::Cell_get;
    pub use crate::data::cell::Cell_new;
    pub use crate::data::cell::Cell_set;

    pub use crate::data::string::Str;
    pub use crate::data::string::Str_clear;
    pub use crate::data::string::Str_concat;
    pub use crate::data::string::Str_eq;
    pub use crate::data::string::Str_from_i32;
    pub use crate::data::string::Str_from_str;
    pub use crate::data::string::Str_insert_char;
    pub use crate::data::string::Str_is_empty;
    pub use crate::data::string::Str_len;
    pub use crate::data::string::Str_new;
    pub use crate::data::string::Str_push_char;
    pub use crate::data::string::Str_push_str;
    pub use crate::data::string::Str_remove;
    pub use crate::data::string::Str_split_off;
    pub use crate::data::string::Str_with_capacity;

    pub use crate::data::primitive::unit;
    pub use crate::data::primitive::Unit;

    pub use crate::runtime::Runtime;

    pub use crate::data::Data;
    pub use crate::db::Database;
    pub use crate::event::Event;
    pub use crate::event::NetEvent;
    pub use crate::event::PipelineEvent;
    pub use crate::event::WorkerEvent;
    pub use crate::iteratee::Iteratee;
    pub use crate::operator::Operator;
    pub use crate::operators::apply::Apply;
    pub use crate::operators::filter::Filter;
    pub use crate::operators::kafka_sink::KafkaSink;
    pub use crate::operators::kafka_source::KafkaSource;
    pub use crate::operators::map::Map;
    pub use crate::operators::scan::Scan;
    pub use crate::operators::shuffle_sink::ShuffleSink;
    pub use crate::operators::shuffle_source::ShuffleSource;
    pub use crate::operators::union::Union;
    pub use crate::operators::worker_source::WorkerSource;
    pub use crate::state::State;

    pub use tokio::select;

    pub use serde::Deserialize;
    pub use serde::Serialize;

    pub use crate::utils;
    pub use std::iter;
    pub use std::sync::Arc;
    pub use std::sync::Mutex;

    pub use std::rc::Rc;
}
