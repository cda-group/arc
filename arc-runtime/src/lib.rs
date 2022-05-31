// #![feature(fn_traits)]
// #![feature(unboxed_closures)]
// #![feature(arbitrary_self_types)]
// #![feature(async_closure)]
// #![feature(async_stream)]
// #![feature(stream_from_iter)]
#![feature(try_trait_v2)]
#![feature(type_alias_impl_trait)]
#![feature(once_cell)]
#![feature(never_type)]
#![feature(new_uninit)]
#![allow(unused)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_doctest_main)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::len_without_is_empty)]

pub mod context;
pub mod control;
pub mod data;
// pub mod operators;
pub mod dispatch;
pub mod manager;
pub mod runtime;
pub mod timer;

pub mod prelude {
    // Data types
    pub use crate::context::Context;
    pub use crate::control::Control;
    pub use crate::control::Control::Continue;
    pub use crate::control::Control::Finished;
    pub use crate::data::channels::channel;
    pub use crate::data::channels::Endpoint;
    pub use crate::data::channels::PullChan;
    pub use crate::data::channels::PushChan;
    // pub use crate::data::erased::Erased;
    pub use crate::data::gc::Gc;
    pub use crate::data::gc::Heap;
    pub use crate::data::gc::Trace;
    pub use crate::data::primitives::bool;
    pub use crate::data::primitives::char;
    pub use crate::data::primitives::f32;
    pub use crate::data::primitives::f64;
    pub use crate::data::primitives::i128;
    pub use crate::data::primitives::i16;
    pub use crate::data::primitives::i32;
    pub use crate::data::primitives::i64;
    pub use crate::data::primitives::i8;
    pub use crate::data::primitives::u128;
    pub use crate::data::primitives::u16;
    pub use crate::data::primitives::u32;
    pub use crate::data::primitives::u64;
    pub use crate::data::primitives::u8;
    pub use crate::data::primitives::unit;
    pub use crate::data::primitives::Unit;
    pub use crate::data::serde::deserialise;
    pub use crate::data::serde::serialise;
    pub use crate::data::serde::SerdeState;
    pub use crate::data::Data;
    pub use crate::data::Key;
    pub use crate::dispatch::spawn;
    pub use crate::dispatch::Execute;
    pub use crate::manager::Manager;
    pub use crate::manager::TaskMessage;
    pub use crate::runtime::Runtime;

    pub use crate::data::primitives::bool_assert;
    pub use crate::data::primitives::Str_panic;
    pub use crate::data::primitives::Str_print;

    pub use crate::data::strings::Str;
    pub use crate::data::strings::Str_clear;
    pub use crate::data::strings::Str_concat;
    pub use crate::data::strings::Str_eq;
    pub use crate::data::strings::Str_from_i32;
    pub use crate::data::strings::Str_from_str;
    pub use crate::data::strings::Str_insert_char;
    pub use crate::data::strings::Str_is_empty;
    pub use crate::data::strings::Str_len;
    pub use crate::data::strings::Str_new;
    pub use crate::data::strings::Str_push_char;
    pub use crate::data::strings::Str_push_str;
    pub use crate::data::strings::Str_remove;
    pub use crate::data::strings::Str_split_off;
    pub use crate::data::strings::Str_with_capacity;

    pub use crate::data::vectors::Vector;
    pub use crate::data::vectors::Vector_capacity;
    pub use crate::data::vectors::Vector_clear;
    pub use crate::data::vectors::Vector_dedup;
    pub use crate::data::vectors::Vector_get;
    pub use crate::data::vectors::Vector_insert;
    pub use crate::data::vectors::Vector_is_empty;
    pub use crate::data::vectors::Vector_iterator;
    pub use crate::data::vectors::Vector_len;
    pub use crate::data::vectors::Vector_new;
    pub use crate::data::vectors::Vector_pop;
    pub use crate::data::vectors::Vector_push;
    pub use crate::data::vectors::Vector_remove;
    pub use crate::data::vectors::Vector_with_capacity;

    pub use crate::data::vectors::VectorIterator_is_empty;
    pub use crate::data::vectors::VectorIterator_new;
    pub use crate::data::vectors::VectorIterator_next;

    pub use crate::data::cells::Cell;
    pub use crate::data::cells::Cell_get;
    pub use crate::data::cells::Cell_new;
    pub use crate::data::cells::Cell_set;

    pub use crate::data::channels::PullChan_pull;
    pub use crate::data::channels::PushChan_push;

    // Declarative macros
    pub use crate::declare;
    pub use crate::function;

    // Procedural macros
    pub use macros::access;
    pub use macros::call;
    pub use macros::call_async;
    pub use macros::call_async_indirect;
    pub use macros::call_indirect;
    pub use macros::enwrap;
    pub use macros::erase;
    pub use macros::is;
    pub use macros::new;
    pub use macros::rewrite;
    pub use macros::serde_state;
    pub use macros::spawn;
    pub use macros::transition;
    pub use macros::unerase;
    pub use macros::unwrap;
    pub use macros::val;
    pub use macros::vector;
    pub use macros::NoDebug;
    pub use macros::NoSerde;
    pub use macros::NoTrace;
    pub use macros::Send;
    pub use macros::Sync;
    pub use macros::Trace;
    pub use macros::Unpin;

    // Re-exports
    pub use kompact::prelude::info;
    pub use kompact::prelude::warn;
    pub use kompact::prelude::Actor;
    pub use kompact::prelude::ActorRaw;
    pub use kompact::prelude::ActorRefFactory;
    pub use kompact::prelude::Component;
    pub use kompact::prelude::ComponentContext;
    pub use kompact::prelude::ComponentDefinition;
    pub use kompact::prelude::ComponentDefinitionAccess;
    pub use kompact::prelude::ComponentLifecycle;
    pub use kompact::prelude::ComponentLogging;
    pub use kompact::prelude::DeadletterBox;
    pub use kompact::prelude::DynamicPortAccess;
    pub use kompact::prelude::ExecuteResult;
    pub use kompact::prelude::Handled;
    pub use kompact::prelude::KompactConfig;
    pub use kompact::prelude::KompactSystem;
    pub use kompact::prelude::MsgEnvelope;
    pub use kompact::prelude::NetMessage;
    pub use kompact::prelude::NetworkConfig;
    pub use kompact::prelude::Never;
    pub use kompact::prelude::SystemHandle;

    pub use derive_more::Constructor as New;
    pub use derive_more::Deref;
    pub use derive_more::DerefMut;
    pub use derive_more::From;

    pub use time::macros::date;
    pub use time::macros::time;
    pub use time::Duration;
    pub use time::PrimitiveDateTime as DateTime;

    pub use futures::executor::block_on;
    pub use futures::future::BoxFuture;
    pub use futures::future::FutureExt;

    pub use replace_with::replace_with_or_abort_and_return;

    pub use hexf::hexf32;
    pub use hexf::hexf64;

    pub use ::serde_state;
    pub use serde::Deserializer;
    pub use serde::Serializer;
    pub use serde_derive_state::DeserializeState;
    pub use serde_derive_state::SerializeState;
    pub use serde_state::DeserializeState;
    pub use serde_state::SerializeState;

    pub use std::any::Any;
    pub use std::any::TypeId;
    pub use std::cell::UnsafeCell;
    pub use std::fmt::Debug;
    pub use std::future::Future;
    pub use std::hash::Hash;
    pub use std::pin::Pin;
    pub use std::sync::Arc;
    pub use std::task::Context as PollContext;
    pub use std::task::Poll;
    pub use std::task::Poll::Pending;
    pub use std::task::Poll::Ready;

    pub use tokio::runtime::Builder as RuntimeBuilder;
    pub use tokio::task::spawn_local;
}
