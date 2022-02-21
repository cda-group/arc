use crate::prelude::*;

pub mod sharable {
    use crate::prelude::*;

    #[derive(Clone, From, Deref, DerefMut, Debug, Collectable, Finalize, Send, Sync, Unpin)]
    #[from(forward)]
    pub struct String(pub Gc<ConcreteString>);

    pub type ConcreteString = comet::alloc::string::String<Immix>;

    impl Alloc<String> for ConcreteString {
        fn alloc(self, ctx: Context) -> String {
            String(ctx.mutator().allocate(self, AllocationSpace::New).into())
        }
    }

    unsafe impl Trace for String {
        fn trace(&mut self, vis: &mut dyn Visitor) {
            self.0.trace(vis)
        }
    }
}

mod sendable {
    use crate::prelude::*;

    #[derive(Clone, From, Send, Serialize, Deserialize)]
    #[from(forward)]
    pub struct String(pub ConcreteString);

    pub type ConcreteString = Box<str>;
}

impl DynSharable for sharable::String {
    type T = sendable::String;
    fn into_sendable(&self, ctx: Context) -> Self::T {
        self.0.to_string().into()
    }
}

impl DynSendable for sendable::String {
    type T = sharable::String;
    fn into_sharable(&self, ctx: Context) -> Self::T {
        String::from_str(self.0.as_ref(), ctx)
    }
}

pub use sharable::String;

impl String {
    pub fn new(ctx: Context) -> String {
        sharable::ConcreteString::new(ctx.mutator()).alloc(ctx)
    }

    pub fn with_capacity(capacity: usize, ctx: Context) -> String {
        sharable::ConcreteString::with_capacity(ctx.mutator(), capacity).alloc(ctx)
    }

    pub fn push(mut self, ch: char, ctx: Context) {
        self.0.push(ctx.mutator(), ch)
    }

    pub fn push_str(mut self, s: &str, ctx: Context) {
        self.0.push_str(ctx.mutator(), s)
    }

    pub fn from_str(s: &str, ctx: Context) -> String {
        let mut new = sharable::ConcreteString::with_capacity(ctx.mutator(), s.len());
        new.push_str(ctx.mutator(), s);
        new.alloc(ctx)
    }

    pub fn remove(&mut self, idx: usize, _: Context) -> char {
        self.0.remove(idx)
    }

    pub fn insert(&mut self, idx: usize, ch: char, ctx: Context) {
        self.0.insert(ctx.mutator(), idx, ch)
    }

    pub fn is_empty(&mut self, _: Context) -> bool {
        self.0.is_empty()
    }

    pub fn split_off(&mut self, at: usize, ctx: Context) -> String {
        self.0.split_off(ctx.mutator(), at).alloc(ctx)
    }

    pub fn clear(&mut self, _: Context) {
        self.0.clear()
    }

    pub fn len(&self, _: Context) -> usize {
        self.0.len()
    }
}
