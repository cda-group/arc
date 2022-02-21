use kompact::prelude::*;
use std::marker::PhantomData;
use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::Sender;

use crate::control::Control;
use crate::data::Sharable;

use crate::prelude::*;

#[derive(Collectable, Finalize, NoTrace, NoSerde, NoDebug)]
pub struct Pushable<T: Sharable>(Sender<T::T>);

impl<T: Sharable> Clone for Pushable<T> {
    fn clone(&self) -> Self {
        Pushable(self.0.clone())
    }
}

#[derive(Collectable, Finalize, NoTrace, NoSerde, NoDebug)]
pub struct Pullable<T: Sharable>(Sender<T::T>, Receiver<T::T>);

impl<T: Sharable> Clone for Pullable<T> {
    fn clone(&self) -> Self {
        Pullable(self.0.clone(), self.0.subscribe())
    }
}

crate::data::convert_reflexive!({T: Sharable} Pushable<T>);
crate::data::convert_reflexive!({T: Sharable} Pullable<T>);

crate::data::channels::impl_channel!();

/// TODO: Processing will currently only stop if all pullers are dropped.
pub fn channel<T: Sharable>(_: Context) -> (Pushable<T>, Pullable<T>)
where
    T::T: Sendable,
{
    let (l, r) = tokio::sync::broadcast::channel(100);
    (Pushable(l.clone()), Pullable(l, r))
}

impl<T: Sharable> Pushable<T> {
    pub async fn push(&self, data: T, ctx: Context) -> Control<()> {
        self.0
            .send(data.into_sendable(ctx))
            .map(|_| Control::Continue(()))
            .unwrap_or(Control::Finished)
    }
}

impl<T: Sharable> Pullable<T> {
    pub async fn pull(&mut self, ctx: Context) -> Control<<T::T as DynSendable>::T> {
        self.1
            .recv()
            .await
            .map(|v| Control::Continue(v.into_sharable(ctx)))
            .unwrap_or(Control::Finished)
    }
}
