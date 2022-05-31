use crate::prelude::*;

#[derive(Copy, Clone, From, Deref, DerefMut, Debug, Trace, Send, Sync, Unpin)]
#[from(forward)]
// #[serde_state]
pub struct Str(pub Gc<String>);

impl Trace for String {
    fn trace(&self, heap: Heap) {}

    fn root(&self, heap: Heap) {}

    fn unroot(&self, heap: Heap) {}

    fn copy(&self, heap: Heap) -> Self {
        self.clone()
    }
}

#[rewrite]
impl Str {
    pub fn new(ctx: Context<impl Execute>) -> Str {
        Str(ctx.heap.allocate(std::string::String::new()))
    }

    pub fn with_capacity(capacity: usize, ctx: Context<impl Execute>) -> Str {
        Str(ctx
            .heap
            .allocate(std::string::String::with_capacity(capacity)))
    }

    pub fn push_char(mut self, ch: char, ctx: Context<impl Execute>) {
        self.0.push(ch)
    }

    pub fn push_str(mut self, s: &str, ctx: Context<impl Execute>) {
        self.0.push_str(s)
    }

    pub fn from_str(s: &str, ctx: Context<impl Execute>) -> Str {
        let mut new = Str::with_capacity(s.len(), ctx);
        new.push_str(s, ctx);
        new
    }

    pub fn remove(mut self, idx: u32, _ctx: Context<impl Execute>) -> char {
        self.0.remove(idx as usize)
    }

    pub fn insert_char(mut self, idx: u32, ch: char, ctx: Context<impl Execute>) {
        self.0.insert(idx as usize, ch)
    }

    pub fn is_empty(mut self, _ctx: Context<impl Execute>) -> bool {
        self.0.is_empty()
    }

    pub fn split_off(mut self, at: u32, ctx: Context<impl Execute>) -> Str {
        Str(ctx.heap.allocate(self.0.split_off(at as usize)))
    }

    pub fn clear(mut self, _ctx: Context<impl Execute>) {
        self.0.clear()
    }

    pub fn len(self, _ctx: Context<impl Execute>) -> u32 {
        self.0.len() as u32
    }

    pub fn from_i32(i: i32, ctx: Context<impl Execute>) -> Str {
        let mut new = Str::new(ctx);
        new.0.push_str(&i.to_string());
        new
    }

    pub fn eq(self, other: Str, ctx: Context<impl Execute>) -> bool {
        self.0.eq(&other.0)
    }

    pub fn concat(self, other: Str, ctx: Context<impl Execute>) -> Str {
        let mut new = Str::new(ctx);
        new.0.push_str(&self.0);
        new.0.push_str(&other.0);
        new
    }
}
