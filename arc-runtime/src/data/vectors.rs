use crate::prelude::*;

#[derive(Copy, Clone, From, Debug, Send, Sync, Unpin, Trace)]
#[from(forward)]
#[serde_state]
pub struct Vector<T: Data>(pub Gc<Vec<T>>);

impl<T: Trace> Trace for Vec<T> {
    fn trace(&self, heap: Heap) {
        self.iter().for_each(|item| item.trace(heap));
    }
    fn root(&self, heap: Heap) {
        self.iter().for_each(|item| item.root(heap));
    }
    fn unroot(&self, heap: Heap) {
        self.iter().for_each(|item| item.unroot(heap));
    }
    fn copy(&self, heap: Heap) -> Self {
        self.iter().map(|v| v.copy(heap)).collect::<Vec<_>>()
    }
}

impl<T: Data> Vector<T> {
    pub fn retain<F>(mut self, f: F, ctx: Context<impl Execute>)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f);
    }
    pub fn as_slice_mut(&mut self, ctx: Context<impl Execute>) -> &mut [T] {
        self.0.as_mut_slice()
    }

    pub fn dedup(mut self, ctx: Context<impl Execute>)
    where
        T: PartialEq,
    {
        self.0.dedup();
    }

    pub fn shrink_to(mut self, min_capacity: usize, ctx: Context<impl Execute>) {
        self.0.shrink_to(min_capacity);
    }

    pub fn resize(mut self, new_len: usize, value: T, ctx: Context<impl Execute>) {
        self.0.resize(new_len, value);
    }
}

impl<T: Data> std::ops::Deref for Vector<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.0.deref()
    }
}

impl<T: Data> std::ops::DerefMut for Vector<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.0.deref_mut()
    }
}

#[rewrite]
impl<T: Data> Vector<T> {
    pub fn iterator(self, ctx: Context<impl Execute>) -> VectorIterator<T> {
        VectorIterator::new(self)
    }

    pub fn new(ctx: Context<impl Execute>) -> Vector<T> {
        Vector(ctx.heap.allocate(Vec::<T>::new()))
    }

    pub fn from_vec(vec: Vec<T>, ctx: Context<impl Execute>) -> Vector<T> {
        Vector(ctx.heap.allocate(vec))
    }

    pub fn with_capacity(capacity: usize, ctx: Context<impl Execute>) -> Vector<T> {
        Vector(ctx.heap.allocate(Vec::<T>::with_capacity(capacity)))
    }

    pub fn capacity(self, ctx: Context<impl Execute>) -> usize {
        self.0.capacity()
    }

    pub fn len(self, ctx: Context<impl Execute>) -> usize {
        self.0.len()
    }

    pub fn clear(mut self, ctx: Context<impl Execute>) {
        self.0.clear();
    }

    pub fn push(mut self, value: T, ctx: Context<impl Execute>) {
        self.0.push(value);
    }

    pub fn pop(mut self, ctx: Context<impl Execute>) -> Option<T> {
        self.0.pop()
    }

    pub fn remove(mut self, index: usize, ctx: Context<impl Execute>) -> T {
        self.0.remove(index)
    }

    pub fn get(self, index: usize, ctx: Context<impl Execute>) -> T {
        self.0[index]
    }

    pub fn insert(mut self, index: usize, value: T, ctx: Context<impl Execute>) {
        self.0.insert(index, value);
    }

    pub fn is_empty(self, ctx: Context<impl Execute>) -> bool {
        self.0.is_empty()
    }
}

#[allow(non_snake_case)]
pub fn Vector_dedup<T: Data>(self_param: Vector<T>, ctx: Context<impl Execute>)
where
    T: PartialEq,
{
    Vector::dedup(self_param, ctx)
}

#[derive(Copy, Clone, From, Debug, Send, Sync, Unpin, Trace)]
#[from(forward)]
#[serde_state]
pub struct VectorIterator<T: Data> {
    vec: Vector<T>,
    offset: u64,
}

#[rewrite]
impl<T: Data> VectorIterator<T> {
    fn new(vec: Vector<T>) -> VectorIterator<T> {
        VectorIterator { vec, offset: 0 }
    }

    pub fn is_empty(self, ctx: Context<impl Execute>) -> bool {
        self.offset >= self.vec.len(ctx) as u64
    }

    pub fn next(mut self, ctx: Context<impl Execute>) -> T {
        let index = self.offset as usize;
        self.offset += 1;
        self.vec.get(index, ctx)
    }
}
