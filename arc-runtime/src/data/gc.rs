use crate::data::serde::Serde;
use crate::data::Data;
use crate::prelude::serde_state;
use crate::prelude::Context;
use crate::prelude::Execute;
use crate::prelude::Send;
use crate::prelude::Sync;
use crate::prelude::Unpin;
use serde::de::Deserializer;
use serde::ser::SerializeStruct;
use serde::ser::Serializer;
use serde_derive_state::DeserializeState;
use serde_derive_state::SerializeState;
use serde_state::ser::Seeded;
use serde_state::DeserializeState;
use serde_state::SerializeState;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::marker::PhantomData;

#[derive(Send, Sync, Unpin)]
pub struct Gc<T: Trace>(*mut T);

#[derive(Copy, Clone)]
struct DynGc(*mut dyn Trace);

impl<T: Trace> From<usize> for Gc<T> {
    fn from(ptr: usize) -> Gc<T> {
        Gc(ptr as *mut T)
    }
}

impl<T: Trace> Gc<T> {
    /// Create a new garbage collected value.
    fn new(value: T) -> Self {
        Self(Box::into_raw(Box::new(value)))
    }

    fn erase(&self) -> DynGc {
        DynGc(self.0 as *mut dyn Trace)
    }

    fn id(&self) -> usize {
        self.0 as usize
    }

    fn get(&self) -> &T {
        unsafe { &*self.0 }
    }

    fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}

impl DynGc {
    fn id(&self) -> usize {
        self.0 as *const () as usize
    }

    fn get(&self) -> &dyn Trace {
        unsafe { &*self.0 }
    }

    fn get_mut(&mut self) -> &mut dyn Trace {
        unsafe { &mut *self.0 }
    }

    fn unerase<T: Trace>(self) -> Gc<T> {
        Gc(self.0 as *mut T)
    }
}

/// Types that implement this can be garbage collected.
pub trait Trace: 'static {
    fn trace(&self, heap: Heap);
    fn root(&self, heap: Heap);
    fn unroot(&self, heap: Heap);
    fn copy(&self, heap: Heap) -> Self
    where
        Self: Sized;
}

macro_rules! impl_primitive {
    ($($($ty:ty)*),*) => {
        $(
            impl Trace for $($ty)* {
                fn trace(&self, _: Heap) {}
                fn root(&self, _: Heap) {}
                fn unroot(&self, _: Heap) {}
                fn copy(&self, _: Heap) -> Self {
                    *self
                }
            }
        )*
    };
}

impl_primitive!(
    (),
    i8,
    i16,
    i32,
    i64,
    i128,
    u8,
    u16,
    u32,
    u64,
    u128,
    f32,
    f64,
    bool,
    char,
    &'static str
);

macro_rules! impl_tuple {
    (
        $($id:ident),+
    ) => {
        #[allow(non_snake_case)]
        impl<$($id: Trace,)+> Trace for ($($id,)+) {
            fn trace(&self, heap: Heap) {
                let ($($id,)+) = self;
                $($id.trace(heap);)+
            }
            fn root(&self, heap: Heap) {
                let ($($id,)+) = self;
                $($id.root(heap);)+
            }
            fn unroot(&self, heap: Heap) {
                let ($($id,)+) = self;
                $($id.unroot(heap);)+
            }
            fn copy(&self, heap: Heap) -> Self {
                let ($($id,)+) = self;
                ($($id.copy(heap),)+)
            }
        }
    };
}

impl_tuple!(A);
impl_tuple!(A, B);
impl_tuple!(A, B, C);
impl_tuple!(A, B, C, D);
impl_tuple!(A, B, C, D, E);

impl<T: Trace> Trace for Gc<T> {
    fn trace(&self, heap: Heap) {
        if let Some(object) = heap.get_mut().old_objects.remove(&self.id()) {
            heap.get_mut().new_objects.insert(self.id(), object);
            self.get().trace(heap);
        }
    }
    fn root(&self, heap: Heap) {
        heap.get_mut().roots.insert(self.id(), self.erase());
    }
    fn unroot(&self, heap: Heap) {
        heap.get_mut().roots.remove(&self.id());
    }
    fn copy(&self, heap: Heap) -> Self {
        if let Some(obj) = heap.get().copied.get(&self.id()) {
            obj.unerase()
        } else {
            let mut object: Self = Gc::from(Box::into_raw(Box::<Gc<T>>::new_uninit()) as usize);
            heap.get_mut().copied.insert(self.id(), object.erase());
            heap.get_mut()
                .old_objects
                .insert(object.id(), object.erase());
            unsafe {
                *object = (*self.0).copy(heap);
            }
            object
        }
    }
}

impl Trace for DynGc {
    fn trace(&self, heap: Heap) {
        if let Some(object) = heap.get_mut().old_objects.remove(&self.id()) {
            heap.get_mut().new_objects.insert(self.id(), object);
            self.get().trace(heap);
        }
    }
    fn root(&self, heap: Heap) {
        heap.get_mut().roots.insert(self.id(), *self);
    }
    fn unroot(&self, heap: Heap) {
        heap.get_mut().roots.remove(&self.id());
    }
    fn copy(&self, _: Heap) -> Self {
        unreachable!("DynGc cannot be copied. This error cannot happen.");
    }
}

/// Heap for storing objects.
#[derive(Send, Sync, Copy, Clone, Debug)]
pub struct Heap(*mut Core);

impl Trace for Heap {
    fn trace(&self, heap: Heap) {}
    fn root(&self, heap: Heap) {}
    fn unroot(&self, heap: Heap) {}
    fn copy(&self, heap: Heap) -> Self {
        unreachable!("Heap cannot be copied. This error cannot happen.");
    }
}

impl std::ops::Deref for Heap {
    type Target = Core;

    fn deref(&self) -> &Core {
        unsafe { &*self.0 }
    }
}

impl std::ops::DerefMut for Heap {
    fn deref_mut(&mut self) -> &mut Core {
        unsafe { &mut *self.0 }
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self(Box::into_raw(Box::new(Core::default())))
    }
}

#[derive(Default)]
pub struct Core {
    roots: HashMap<usize, DynGc>,
    old_objects: HashMap<usize, DynGc>,
    new_objects: HashMap<usize, DynGc>,
    copied: HashMap<usize, DynGc>,
}

impl Heap {
    pub fn new() -> Self {
        Self::default()
    }

    fn get(&self) -> &Core {
        unsafe { &*self.0 }
    }

    #[allow(clippy::mut_from_ref)]
    fn get_mut(&self) -> &mut Core {
        unsafe { &mut *self.0 }
    }

    /// Allocate a new object.
    pub fn allocate<T: Trace>(&self, value: T) -> Gc<T> {
        let object = Gc::new(value);
        self.get_mut()
            .old_objects
            .insert(object.id(), object.erase());
        object
    }

    /// Reallocate an object from potentially another heap to this heap.
    pub fn reallocate<T: Trace>(self, object: Gc<T>) -> Gc<T> {
        self.get_mut().copied.clear();
        object.copy(self)
    }

    /// Run the garbage collector.
    pub fn collect(self) {
        // TODO: Fix potential UB by passing a tracer instead of self
        for root in self.get().roots.values() {
            root.trace(self);
        }
        for (_, object) in self.get_mut().old_objects.drain() {
            unsafe {
                drop(Box::from_raw(object.0));
            }
        }
        let heap = self.get_mut();
        std::mem::swap(&mut heap.old_objects, &mut heap.new_objects);
    }

    pub fn destroy(self) {
        self.collect();
    }
}
use std::sync::Arc;

/// Arc<T> is bound to no heap.
impl<T: 'static> Trace for Arc<T> {
    fn trace(&self, heap: Heap) {}

    fn root(&self, heap: Heap) {}

    fn unroot(&self, heap: Heap) {}

    fn copy(&self, heap: Heap) -> Self {
        self.clone()
    }
}

// Convenience stuff

impl<T: Trace> Copy for Gc<T> {}

impl<T: Trace> Clone for Gc<T> {
    fn clone(&self) -> Gc<T> {
        *self
    }
}

impl<T: Trace + Eq> Eq for Gc<T> {}

impl<T: Trace + PartialEq> PartialEq for Gc<T> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { *self.0 == *other.0 }
    }
}

impl<T: Trace + std::fmt::Debug> std::fmt::Debug for Gc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { write!(f, "{:?}", &*self.0) }
    }
}

impl<T: Trace> std::ops::Deref for Gc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: Trace> std::ops::DerefMut for Gc<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl std::ops::Deref for DynGc {
    type Target = dyn Trace;
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl std::ops::DerefMut for DynGc {
    fn deref_mut(&mut self) -> &mut dyn Trace {
        self.get_mut()
    }
}

impl<T: 'static> Trace for std::marker::PhantomData<T> {
    fn trace(&self, heap: Heap) {}

    fn root(&self, heap: Heap) {}

    fn unroot(&self, heap: Heap) {}

    fn copy(&self, heap: Heap) -> Self {
        *self
    }
}

use crate::data::serde::SerdeState;

#[serde_state]
struct Node<T: Serde> {
    id: usize,
    data: Option<T>,
}

impl<T: Serde + Trace> SerializeState<SerdeState> for Gc<T> {
    fn serialize_state<S: Serializer>(&self, s: S, state: &SerdeState) -> Result<S::Ok, S::Error> {
        let mut s = s.serialize_struct("Gc", 2)?;
        s.serialize_field("id", &Seeded::new(state, self.id()))?;
        let data = if state.serialized().insert(self.id()) {
            Some(Seeded::new(state, self.get()))
        } else {
            None
        };
        s.serialize_field("data", &data)?;
        s.end()
    }
}

impl<'de, T: Serde + Trace> DeserializeState<'de, SerdeState> for Gc<T> {
    fn deserialize_state<D: Deserializer<'de>>(
        ctx: &mut SerdeState,
        d: D,
    ) -> Result<Self, D::Error> {
        let node = Node::deserialize_state(ctx, d)?;
        let id = ctx
            .deserialized
            .entry(node.id)
            .or_insert_with(|| Box::into_raw(Box::<Gc<T>>::new_uninit()) as usize);
        let id = *id;
        let mut gc: Gc<T> = Gc::from(id);
        if let Some(data) = node.data {
            *gc.get_mut() = data;
            ctx.heap.get_mut().old_objects.insert(id, gc.erase()); // Move to current heap
        }
        Ok(gc)
    }
}
