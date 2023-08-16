use serde::Deserialize;
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;

pub trait Data:
    DeepClone + Clone + Send + Serialize + for<'a> Deserialize<'a> + Unpin + Debug + 'static
{
}
impl<T> Data for T where
    T: DeepClone + Clone + Send + Serialize + for<'a> Deserialize<'a> + Unpin + Debug + 'static
{
}

pub trait Key: Data + Eq + PartialEq + Hash {}
impl<T> Key for T where T: Data + Eq + PartialEq + Hash {}

pub trait DeepClone: Clone {
    fn deep_clone(&self) -> Self;
}

macro_rules! impl_deep_clone_tuple {
    { $h:ident $(, $t:ident)* } => {
        impl<$h: DeepClone, $($t: DeepClone),*> DeepClone for ($h, $($t,)*) {
            #[allow(non_snake_case)]
            fn deep_clone(&self) -> Self {
                let ($h, $($t,)*) = self;
                ($h.deep_clone(), $($t.deep_clone(),)*)
            }
        }
        impl_deep_clone_tuple! { $($t),* }
    };
    {} => {}
}

impl_deep_clone_tuple!(A, B, C, D, E, F, G, H);

impl<T: DeepClone> DeepClone for std::rc::Rc<T> {
    fn deep_clone(&self) -> Self {
        std::rc::Rc::new(self.as_ref().deep_clone())
    }
}

impl<T: DeepClone> DeepClone for std::sync::Arc<T> {
    fn deep_clone(&self) -> Self {
        std::sync::Arc::new(self.as_ref().deep_clone())
    }
}

impl<T: DeepClone> DeepClone for std::vec::Vec<T> {
    fn deep_clone(&self) -> Self {
        self.iter().map(|x| x.deep_clone()).collect()
    }
}

macro_rules! impl_deep_clone_scalar {
    { $t:ty } => {
        impl DeepClone for $t {
            fn deep_clone(&self) -> Self {
                *self
            }
        }
    };
}

impl_deep_clone_scalar! { () }
impl_deep_clone_scalar! { bool }
impl_deep_clone_scalar! { i8 }
impl_deep_clone_scalar! { i16 }
impl_deep_clone_scalar! { i32 }
impl_deep_clone_scalar! { i64 }
impl_deep_clone_scalar! { i128 }
impl_deep_clone_scalar! { isize }
impl_deep_clone_scalar! { u8 }
impl_deep_clone_scalar! { u16 }
impl_deep_clone_scalar! { u32 }
impl_deep_clone_scalar! { u64 }
impl_deep_clone_scalar! { u128 }
impl_deep_clone_scalar! { usize }
impl_deep_clone_scalar! { f32 }
impl_deep_clone_scalar! { f64 }
impl_deep_clone_scalar! { char }
impl_deep_clone_scalar! { &'static str }
