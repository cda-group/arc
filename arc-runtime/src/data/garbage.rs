use comet::api::Collectable;
use comet::api::Finalize;
use comet::api::Trace;
use comet::immix::Immix;

use derive_more::AsRef;

use crate::prelude::*;

pub trait Garbage: Collectable + Trace + Finalize {}
impl<T> Garbage for T where T: Collectable + Trace + Finalize {}

#[derive(AsRef, Deref, DerefMut, From, Debug)]
#[deref(forward)]
#[deref_mut(forward)]
#[as_ref(forward)]
pub struct Gc<T: Garbage>(comet::api::Gc<T, Immix>);

impl<T: Garbage> Clone for Gc<T> {
    fn clone(&self) -> Self {
        self.0.into()
    }
}

pub trait Alloc<T> {
    fn alloc(self, ctx: Context) -> T;
}

macro_rules! alloc_identity {
    { $ty:ty } => {
        impl Alloc<$ty> for $ty {
            #[inline(always)]
            fn alloc(self, ctx: Context) -> $ty {
                self
            }
        }
    }
}

pub(crate) use alloc_identity;
