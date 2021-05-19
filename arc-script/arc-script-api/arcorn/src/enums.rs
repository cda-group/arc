//! Macros for using arbitrary enums without having to
//! implement a bunch of boilerplate methods.
//!
//! Requirements which must be satisfied by arc-script:
//! * Each enum variant has exactly one field.
//! * Each enum variant identifier is globally unique.
//!
//! All of these assumptions will be ensured by the Arcorn interface.

/// Enwraps a value into an enum-variant.
///
/// ```
/// #[arc_script::arcorn::rewrite]
/// enum Foo {
///     Bar(i32),
///     Baz(i32)
/// }
/// let foo = arc_script::arcorn::enwrap!(Bar, 5);
/// ```
#[macro_export]
macro_rules! enwrap {
    ($variant:ident , $expr:expr) => {
        ($variant($expr)).wrap()
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arcorn::paste! {
            ([<Enum $enum>]::$variant($expr)).wrap()
        }
    };
}

/// Returns `true` if enum is a certain variant, else `false`.
///
/// ```
/// #[arc_script::arcorn::rewrite]
/// enum Foo {
///     Bar(i32),
///     Baz(i32)
/// }
///
/// let foo = arc_script::arcorn::enwrap!(Bar, 5);
/// assert!(arc_script::arcorn::is!(Bar, foo));
/// ```
#[macro_export]
macro_rules! is {
    ($variant:ident , $expr:expr) => {
        if let Some($variant(_)) = &$expr.this {
            true
        } else {
            false
        }
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arcorn::paste! {
            if let Some([<Enum $enum>]::$variant(_)) = &$expr.this {
                true
            } else {
                false
            }
        }
    };
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// #[arc_script::arcorn::rewrite]
/// enum FooEnum {
///     Bar(i32),
///     Baz(i32)
/// }
///
/// let foo = arc_script::arcorn::enwrap!(Bar, 5);
/// let bar = arc_script::arcorn::unwrap!(Bar, foo);
/// ```
#[macro_export]
macro_rules! unwrap {
    ($variant:ident , $expr:expr) => {
        if let Some($variant(v)) = $expr.this {
            v
        } else {
            unreachable!()
        }
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arcorn::paste! {
            if let Some([<Enum $enum>]::$variant(v)) = $expr.this {
                v
            } else {
                unreachable!()
            }
        }
    };
}
