use crate::prelude::*;

pub mod sharable {
    use crate::prelude::*;

    #[derive(
        Clone, From, Deref, DerefMut, Finalize, Collectable, Debug, Send, Sync, Unpin, Trace,
    )]
    #[from(forward)]
    pub struct Vec<T: Sharable>(pub Gc<ConcreteVec<T>>);

    pub type ConcreteVec<T> = comet::alloc::vector::Vector<T, Immix>;

    impl<T: Sharable> Alloc<Vec<T>> for ConcreteVec<T> {
        fn alloc(self, ctx: Context) -> Vec<T> {
            Vec(ctx.mutator().allocate(self, AllocationSpace::New).into())
        }
    }
}

mod sendable {
    use crate::prelude::*;

    #[derive(Clone, From, Send, Serialize, Deserialize)]
    #[serde(bound = "")]
    #[from(forward)]
    pub struct Vec<T: Sendable>(pub ConcreteVec<T>);

    pub type ConcreteVec<T> = std::vec::Vec<T>;
}

impl<T: Sharable> DynSharable for sharable::Vec<T> {
    type T = sendable::Vec<T::T>;
    fn into_sendable(&self, ctx: Context) -> Self::T {
        self.0
            .iter()
            .map(|v| v.clone().into_sendable(ctx))
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice()
            .into()
    }
}

impl<T: Sendable> DynSendable for sendable::Vec<T> {
    type T = sharable::Vec<T::T>;
    fn into_sharable(&self, ctx: Context) -> Self::T {
        let mut s = Vec::<T::T>::with_capacity(self.0.len(), ctx);
        for v in self.0.iter() {
            let v = v.into_sharable(ctx);
            s.0.push(ctx.mutator(), v);
        }
        s
    }
}

pub use sharable::Vec;

impl<T: Sharable> Vec<T> {
    pub fn new(ctx: Context) -> Self {
        sharable::ConcreteVec::<T>::new(ctx.mutator()).alloc(ctx)
    }

    pub fn with_capacity(capacity: usize, ctx: Context) -> Self {
        sharable::ConcreteVec::<T>::with_capacity(ctx.mutator(), capacity).alloc(ctx)
    }

    pub fn write_barrier(&mut self, ctx: Context) {
        self.0.write_barrier(ctx.mutator())
    }

    pub fn as_slice(&self, ctx: Context) -> &[T] {
        self.0.as_slice()
    }

    pub fn as_slice_mut(&mut self, ctx: Context) -> &mut [T] {
        self.0.as_slice_mut()
    }

    pub fn capacity(self, ctx: Context) -> usize {
        self.0.capacity()
    }

    pub fn len(self, ctx: Context) -> usize {
        self.0.len()
    }

    pub fn shrink_to(mut self, min_capacity: usize, ctx: Context) {
        self.0.shrink_to(ctx.mutator(), min_capacity);
    }

    pub fn retain<F>(mut self, f: F, ctx: Context)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f);
    }

    pub fn clear(mut self, ctx: Context) {
        self.0.clear();
    }

    pub fn resize(mut self, new_len: usize, value: T, ctx: Context) {
        self.0.resize(ctx.mutator(), new_len, value);
    }

    pub fn push(mut self, value: T, ctx: Context) {
        self.0.push(ctx.mutator(), value);
    }

    pub fn pop(mut self, ctx: Context) -> Option<T> {
        self.0.pop()
    }

    pub fn remove(mut self, index: usize, ctx: Context) -> T {
        self.0.remove(index)
    }

    pub fn at(self, index: usize, ctx: Context) -> T {
        self.0.at(index).clone()
    }

    pub fn insert(mut self, index: usize, value: T, ctx: Context) {
        self.0.insert(ctx.mutator(), index, value);
    }

    pub fn is_empty(self, ctx: Context) -> bool {
        self.0.is_empty()
    }

    pub fn dedup(mut self, ctx: Context)
    where
        T: PartialEq,
    {
        self.0.dedup();
    }
}
