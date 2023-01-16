#![feature(proc_macro_span)]

use proc_macro::TokenStream;

mod proc_macro_derives;
mod proc_macro_attrs {
    pub mod rewrite {
        pub mod enums;
        pub mod externs;
        pub mod functions;
        pub mod main;
        pub mod structs;
    }
    pub mod export {
        pub mod impls;
    }
}
mod proc_macros;
pub(crate) mod utils;

#[proc_macro_derive(NoTrace)]
pub fn derive_notrace(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_notrace(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Trace)]
pub fn derive_trace(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_trace(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Send)]
pub fn derive_send(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_send(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Sync)]
pub fn derive_sync(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_sync(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Unpin)]
pub fn derive_unpin(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_unpin(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(NoSerde)]
pub fn derive_noserde(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_noserde(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(NoDebug)]
pub fn derive_nodebug(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_nodebug(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro_derive(Accesso)]
pub fn derive_accessor(input: TokenStream) -> TokenStream {
    proc_macro_derives::derive_accessor(syn::parse_macro_input!(input as syn::DeriveInput))
}

#[proc_macro]
pub fn call(input: TokenStream) -> TokenStream {
    proc_macros::call(syn::parse_macro_input!(input as syn::Expr))
}

#[proc_macro]
pub fn call_async(input: TokenStream) -> TokenStream {
    proc_macros::call_async(syn::parse_macro_input!(input as syn::Expr))
}

#[proc_macro]
pub fn call_indirect(input: TokenStream) -> TokenStream {
    proc_macros::call_indirect(syn::parse_macro_input!(input as syn::Expr))
}

/// Enwraps a value into an enum-variant.
///
/// ```
/// use dataflow::prelude::*;
/// mod foo {
///     use dataflow::prelude::*;
///     #[rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
/// let x = enwrap!(foo::Bar::Baz, 5);
/// ```
#[proc_macro]
pub fn enwrap(input: TokenStream) -> TokenStream {
    proc_macros::enwrap(input)
}

/// Returns `true` if enum is a certain variant, else `false`.
///
/// ```
/// use dataflow::prelude::*;
/// mod foo {
///     use dataflow::prelude::*;
///     #[rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = enwrap!(foo::Bar::Baz, 5);
/// assert!(is!(foo::Bar::Baz, x));
/// ```
#[proc_macro]
pub fn is(input: TokenStream) -> TokenStream {
    proc_macros::is(input)
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// use dataflow::prelude::*;
/// mod foo {
///     use dataflow::prelude::*;
///     #[rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = enwrap!(foo::Bar::Baz, 5);
/// let y = unwrap!(foo::Bar::Baz, x);
/// ```
#[proc_macro]
pub fn unwrap(input: TokenStream) -> TokenStream {
    proc_macros::unwrap(input)
}

/// Constructs a struct.
///
/// ```
/// use dataflow::prelude::*;
/// mod foo {
///     use dataflow::prelude::*;
///     #[rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let x = new!(foo::Bar { x: 1, y: 2 });
/// ```
#[proc_macro]
pub fn new(input: TokenStream) -> TokenStream {
    proc_macros::new(input)
}

/// Get the value of a variable.
///
/// ```
/// use dataflow::prelude::*;
/// let a = 5;
/// let b = val!(a);
/// ```
#[proc_macro]
pub fn val(input: TokenStream) -> TokenStream {
    proc_macros::val(input)
}

/// Access a struct's field.
///
/// ```
/// use dataflow::prelude::*;
/// #[rewrite]
/// pub struct Bar {
///     pub x: i32,
///     pub y: i32
/// }
/// fn test() {
///     let a = new!(Bar { x: 0, y: 1 });
///     let b = access!(a, x);
/// }
/// ```
#[proc_macro]
pub fn access(input: TokenStream) -> TokenStream {
    proc_macros::access(input)
}

#[proc_macro]
pub fn get(input: TokenStream) -> TokenStream {
    proc_macros::get(input)
}

#[proc_macro]
pub fn label(input: TokenStream) -> TokenStream {
    proc_macros::label(input)
}

#[proc_macro]
#[allow(non_snake_case)]
pub fn Record(input: TokenStream) -> TokenStream {
    proc_macros::Record(input)
}

#[proc_macro]
#[allow(non_snake_case)]
pub fn Enum(input: TokenStream) -> TokenStream {
    proc_macros::Enum(input)
}

#[proc_macro]
pub fn record(input: TokenStream) -> TokenStream {
    proc_macros::record(input)
}

#[proc_macro]
pub fn vector(input: TokenStream) -> TokenStream {
    proc_macros::vector(input)
}

#[proc_macro_attribute]
pub fn rewrite(attr: TokenStream, input: TokenStream) -> TokenStream {
    let attr = syn::parse_macro_input!(attr as syn::AttributeArgs);
    let item = syn::parse_macro_input!(input as syn::Item);
    let metas = utils::get_metas(&attr);
    match item {
        syn::Item::Enum(item) => proc_macro_attrs::rewrite::enums::rewrite(attr, item),
        syn::Item::Struct(item) => proc_macro_attrs::rewrite::structs::rewrite(attr, item),
        syn::Item::Fn(item) if utils::has_meta_name_val("unmangled", &metas) => {
            proc_macro_attrs::rewrite::externs::rewrite(attr, item)
        }
        syn::Item::Fn(item) if utils::has_meta_key("main", &metas) => {
            proc_macro_attrs::rewrite::main::rewrite(attr, item)
        }
        syn::Item::Fn(item) => proc_macro_attrs::rewrite::functions::rewrite(attr, item),
        _ => panic!("#[rewrite] expects an enum, struct, function, impl, or module as input."),
    }
}

#[proc_macro_attribute]
pub fn export(attr: TokenStream, input: TokenStream) -> TokenStream {
    let attr = syn::parse_macro_input!(attr as syn::AttributeArgs);
    let item = syn::parse_macro_input!(input as syn::Item);
    match item {
        syn::Item::Impl(item) => proc_macro_attrs::export::impls::export(attr, item),
        _ => panic!("#[export] expects a function or impl as input."),
    }
}
