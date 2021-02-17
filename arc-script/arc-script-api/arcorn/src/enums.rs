//! Macros for using arbitrary enums without having to
//! implement a bunch of boilerplate methods.
//!
//! Requirements which must be satisfied by arc-script:
//! * Each enum variant has exactly one field.
//! * Each enum variant identifier is globally unique.
//!
//! All of these assumptions will be ensured by the Arcorn interface.

/// Declares a new enum which is compatible with the `arcorn::{enwrap, unwrap, is}` API.
///
/// Any expansion of the macro satisfies the following properties:
/// * Each enum is wrapped inside a struct (prost requirement).
/// * Each enum implements a method `.wrap()` to wrap it inside the struct.
/// * Each enum variants is imported into the global namespace.
///
/// ```
/// arcorn::declare_enum! {
///     enum Foo {
///         FooBar(i32),
///         FooBaz(i32)
///     }
/// }
/// ```
#[macro_export]
macro_rules! declare_enum {
    {
        enum $name:ident {
            $($variant:ident($ty:ty)),*
        }
    } => {
        paste::paste! {
            pub struct $name {
                this: [<$name Enum>]
            }
            pub enum [<$name Enum>] {
                $($variant($ty)),*
            }
            use [<$name Enum>]::*;
            impl [<$name Enum>] {
                fn wrap(self) -> $name {
                    $name { this: self }
                }
            }
        }
    }
}

/// Enwraps a value into an enum-variant.
///
/// ```
/// arcorn::declare_enum! {
///     enum Foo {
///         FooBar(i32),
///         FooBaz(i32)
///     }
/// }
/// let foo = arcorn::enwrap!(FooBar, 5);
/// ```
#[macro_export]
macro_rules! enwrap {
    {
        $variant:ident , $expr:expr
    } => {
        ($variant($expr)).wrap()
    }
}

/// Returns `true` if enum is a certain variant, else `false`.
///
/// ```
/// arcorn::declare_enum! {
///     enum Foo {
///         FooBar(i32),
///         FooBaz(i32)
///     }
/// }
///
/// let foo = arcorn::enwrap!(FooBar, 5);
/// assert!(arcorn::is!(FooBar, foo));
/// ```
#[macro_export]
macro_rules! is {
    {
        $variant:ident , $expr:expr
    } => {
        if let $variant(_) = &$expr.this {
            true
        } else {
            false
        }
    }
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// arcorn::declare_enum! {
///     enum FooEnum {
///         FooBar(i32),
///         FooBaz(i32)
///     }
/// }
///
/// let foo = arcorn::enwrap!(FooBar, 5);
/// let bar = arcorn::unwrap!(FooBar, foo);
/// ```
#[macro_export]
macro_rules! unwrap {
    {
        $variant:ident , $expr:expr
    } => {
        if let $variant(v) = $expr.this {
            v
        } else {
            unreachable!()
        }
    }
}
