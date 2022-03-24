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

#[rewrite]
impl String {
    pub fn new(ctx: Context) -> String {
        sharable::ConcreteString::new(ctx.mutator()).alloc(ctx)
    }

    pub fn with_capacity(capacity: usize, ctx: Context) -> String {
        sharable::ConcreteString::with_capacity(ctx.mutator(), capacity).alloc(ctx)
    }

    pub fn push_char(mut self, ch: char, ctx: Context) {
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

    pub fn remove(mut self, idx: u32, _ctx: Context) -> char {
        self.0.remove(idx as usize)
    }

    pub fn insert_char(mut self, idx: u32, ch: char, ctx: Context) {
        self.0.insert(ctx.mutator(), idx as usize, ch)
    }

    pub fn is_empty(mut self, _ctx: Context) -> bool {
        self.0.is_empty()
    }

    pub fn split_off(mut self, at: u32, ctx: Context) -> String {
        self.0.split_off(ctx.mutator(), at as usize).alloc(ctx)
    }

    pub fn clear(mut self, _ctx: Context) {
        self.0.clear()
    }

    pub fn len(self, _ctx: Context) -> u32 {
        self.0.len() as u32
    }

    pub fn from_i32(i: i32, ctx: Context) -> String {
        let mut new = String::new(ctx);
        new.0.push_str(ctx.mutator(), &i.to_string());
        new
    }

    pub fn eq(self, other: String, ctx: Context) -> bool {
        self.0.eq(&other.0)
    }

    pub fn concat(self, other: String, ctx: Context) -> String {
        let mut new = String::new(ctx);
        new.0.push_str(ctx.mutator(), &self.0);
        new.0.push_str(ctx.mutator(), &other.0);
        new
    }
}
