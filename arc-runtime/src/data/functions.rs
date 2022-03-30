use crate::context::Context;
use crate::data::DynSharable;
use comet::api::Collectable;
use comet::api::Finalize;
use comet::api::Trace;
use std::fmt::Debug;
use std::ptr::NonNull;

#[macro_export]
macro_rules! declare_functions {
    ($($id:ident),* $(,)?) => {
        #[derive(Send, Sync, Unpin, Collectable, Finalize, NoTrace)]
        pub struct Function<I: 'static, O: 'static> {
            pub ptr: fn(I, Context) -> O,
            pub tag: FunctionTag<I, O>,
        }
        #[derive(Debug, Copy, Send, Sync, Unpin, Serialize, Deserialize)]
        pub struct FunctionTag<I, O>(pub Tag, pub std::marker::PhantomData<(I, O)>);
        #[derive(Debug, Clone, Copy, Send, Serialize, Deserialize)]
        #[allow(non_camel_case_types)]
        pub enum Tag {
            $($id,)*
        }
        impl<I, O> Clone for Function<I, O> {
            fn clone(&self) -> Self {
                Self { ptr: self.ptr.clone(), tag: self.tag.clone() }
            }
        }
        impl<I, O> std::fmt::Debug for Function<I, O> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Ok(())
            }
        }
        impl<I, O> Clone for FunctionTag<I, O> {
            fn clone(&self) -> Self {
                Self(self.0, std::marker::PhantomData)
            }
        }
        impl<I, O> DynSharable for Function<I, O> {
            type T = FunctionTag<I, O>;
            fn into_sendable(&self, ctx: Context) -> Self::T {
                self.tag.clone()
            }
        }
        impl<I: 'static, O: 'static> DynSendable for FunctionTag<I, O> {
            type T = Function<I, O>;
            fn into_sharable(&self, ctx: Context) -> Self::T {
                unsafe {
                    match self.0 {
                        $(Tag::$id => Function {
                            ptr: std::mem::transmute($id as usize),
                            tag: self.clone()
                        }),*
                    }
                }
            }
        }
        impl<I, O> Serialize for Function<I, O> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.tag.0.serialize(serializer)
            }
        }
        impl<'i, I, O> Deserialize<'i> for Function<I, O> {
            fn deserialize<D: Deserializer<'i>>(deserializer: D) -> Result<Self, D::Error> {
                unsafe {
                    match Tag::deserialize(deserializer)? {
                        $(Tag::$id => Ok(Function {
                            ptr: std::mem::transmute($id as usize),
                            tag: FunctionTag(Tag::$id, std::marker::PhantomData)
                        }),)*
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! function {
    // Create a function value
    ($fun:ident) => {
        Function {
            ptr: $fun,
            tag: FunctionTag(Tag::$fun, std::marker::PhantomData),
        }
    };
    // Create a function type
    (($($input:ty),* $(,)?) -> $output:ty) => {
        Function<($($input,)*), $output>
    };
}
