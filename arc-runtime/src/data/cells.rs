use crate::prelude::*;

pub mod sharable {
    use crate::prelude::*;
    #[derive(Clone, From, Finalize, Collectable, Trace, Debug, Send, Sync, Unpin)]
    pub struct Cell<T: Sharable>(pub Gc<T>);
}

pub mod sendable {
    use crate::prelude::*;
    #[derive(Clone, From, Send, Serialize, Deserialize)]
    #[serde(bound = "")]
    pub struct Cell<T: Sendable>(pub T);
}

impl<T: Sharable> DynSharable for sharable::Cell<T> {
    type T = sendable::Cell<T::T>;
    fn into_sendable(&self, ctx: Context) -> Self::T {
        sendable::Cell(self.0.into_sendable(ctx))
    }
}

impl<T: Sendable> DynSendable for sendable::Cell<T> {
    type T = sharable::Cell<T::T>;
    fn into_sharable(&self, ctx: Context) -> Self::T {
        sharable::Cell::new(self.0.into_sharable(ctx), ctx)
    }
}

pub use sharable::Cell;

#[rewrite]
impl<T: Sharable> Cell<T> {
    pub fn new(v: T, ctx: Context) -> Cell<T> {
        Cell(ctx.mutator().allocate(v, AllocationSpace::New).into())
    }
    pub fn get(self, ctx: Context) -> T {
        self.0.inner()
    }
    pub fn set(mut self, v: T, ctx: Context) {
        self.0 = ctx.mutator().allocate(v, AllocationSpace::New).into();
    }
}
