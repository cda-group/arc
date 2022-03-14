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
#![allow(unused)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_doctest_main)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::len_without_is_empty)]

pub mod context;
pub mod control;
pub mod data;
pub mod macros;
pub mod operators;
pub mod runtime;
pub mod task;
pub mod timer;

pub mod prelude {
    // Data types
    pub use crate::context::Context;
    pub use crate::control::Control;
    pub use crate::control::Control::Continue;
    pub use crate::control::Control::Finished;
    pub use crate::data::channels;
    pub use crate::data::channels::Channel;
    pub use crate::data::erased::Erased;
    pub use crate::data::garbage::Alloc;
    pub use crate::data::garbage::Gc;
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
    pub use crate::data::strings::String;
    pub use crate::data::vectors::Vec;
    pub use crate::data::Abstract;
    pub use crate::data::Concrete;
    pub use crate::data::DynSendable;
    pub use crate::data::DynSharable;
    pub use crate::data::Sendable;
    pub use crate::data::Sharable;
    pub use crate::runtime::Runtime;
    pub use crate::task::message::TaskMessage;

    pub use crate::data::primitives::assert;
    pub use crate::data::primitives::panic;
    pub use crate::data::primitives::print;

    // Declarative macros
    pub use crate::access;
    pub use crate::convert_reflexive;
    pub use crate::declare_functions;
    pub use crate::function;
    pub use crate::letroot;
    pub use crate::val;

    // Hidden macros
    pub use crate::_vector;

    // Procedural macros
    pub use macros::call;
    pub use macros::call_indirect;
    pub use macros::enwrap;
    pub use macros::erase;
    pub use macros::is;
    pub use macros::new;
    pub use macros::pull;
    pub use macros::pull_transition;
    pub use macros::push;
    pub use macros::push_transition;
    pub use macros::rewrite;
    pub use macros::terminate;
    pub use macros::transition;
    pub use macros::unerase;
    pub use macros::unwrap;
    pub use macros::vector;
    pub use macros::wait;
    pub use macros::Abstract;
    pub use macros::Alloc;
    pub use macros::Collectable;
    pub use macros::Finalize;
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

    pub use comet::api::Collectable;
    pub use comet::api::Finalize;
    pub use comet::api::Trace;
    pub use comet::api::Visitor;
    pub use comet::gc_base::AllocationSpace;
    pub use comet::immix::instantiate_immix;
    pub use comet::immix::Immix;
    pub use comet::immix::ImmixOptions;
    pub use comet::mopa::TraitObject;
    pub use comet::mutator::MutatorRef;
    pub use comet::shadow_stack::Rootable;
    pub use comet::shadow_stack::Rooted;
    pub use comet::shadow_stack::ShadowStack;
    pub use comet::shadow_stack::ShadowStackInternal;

    pub use derive_more::Constructor as New;
    pub use derive_more::Deref;
    pub use derive_more::DerefMut;
    pub use derive_more::From;

    pub use time::macros::date;
    pub use time::macros::time;
    pub use time::Duration;
    pub use time::PrimitiveDateTime as DateTime;

    pub use futures::future::BoxFuture;
    pub use futures::future::FutureExt;

    pub use replace_with::replace_with_or_abort_and_return;

    pub use hexf::hexf32;
    pub use hexf::hexf64;

    pub use serde::de::DeserializeOwned;
    pub use serde::Deserialize;
    pub use serde::Deserializer;
    pub use serde::Serialize;
    pub use serde::Serializer;
    pub use serde_derive::Deserialize;
    pub use serde_derive::Serialize;

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
}
