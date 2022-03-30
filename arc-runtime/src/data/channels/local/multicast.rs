use async_broadcast::Receiver;
use async_broadcast::Sender;
use kompact::prelude::*;
use std::marker::PhantomData;

use crate::control::Control;
use crate::data::Sharable;

use crate::prelude::*;

#[derive(Clone, Collectable, Finalize, NoTrace, NoSerde, NoDebug)]
pub struct Pushable<T: Sharable>(Sender<T::T>);

#[derive(Clone, Collectable, Finalize, NoTrace, NoSerde, NoDebug)]
pub struct Pullable<T: Sharable>(Receiver<T::T>);

crate::data::convert_reflexive!({T: Sharable} Pushable<T>);
crate::data::convert_reflexive!({T: Sharable} Pullable<T>);

crate::data::channels::impl_channel!();

/// TODO: Processing will currently only stop if all pullers are dropped.
pub fn channel<T: Sharable>(_: Context) -> (Pushable<T>, Pullable<T>)
where
    T::T: Sendable,
{
    let (l, r) = async_broadcast::broadcast(100);
    (Pushable(l), Pullable(r))
}

impl<T: Sharable> Pushable<T> {
    pub async fn push(&self, data: T, ctx: Context) -> Control<()> {
        self.0
            .broadcast(data.into_sendable(ctx))
            .await
            .map(|_| Control::Continue(()))
            .unwrap_or(Control::Finished)
    }
}

impl<T: Sharable> Pullable<T> {
    pub async fn pull(&mut self, ctx: Context) -> Control<<T::T as DynSendable>::T> {
        self.0
            .recv()
            .await
            .map(|v| Control::Continue(v.into_sharable(ctx)))
            .unwrap_or(Control::Finished)
    }
}
