//! Macros for using arbitrary enums without having to
//! implement a bunch of boilerplate methods.
//!
//! Requirements which must be satisfied by arc-script:
//! * Each enum variant has exactly one field.
//! * Each enum variant identifier is globally unique.
//!
//! All of these assumptions will be ensured by the Codegen interface.

/// Enwraps a value into an enum-variant.
///
/// ```
/// mod foo {
///     #[arc_codegen::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
/// let x = arc_codegen::enwrap!(foo::Bar::Baz, 5);
/// ```
#[macro_export]
macro_rules! enwrap {
    (@done $path:path, $expr:expr) => {
        $path($expr).convert()
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::enwrap!(@done $mod::[<Concrete $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::enwrap!(@done [<Concrete $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_codegen::enwrap!(@done $variant, $expr)
    };
}

/// Returns `true` if enum is a certain variant, else `false`.
///
/// ```
/// mod foo {
///     #[arc_codegen::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = arc_codegen::enwrap!(foo::Bar::Baz, 5);
/// assert!(arc_codegen::is!(foo::Bar::Baz, x));
/// ```
#[macro_export]
macro_rules! is {
    (@done $path:path, $expr:expr) => {
        if let $path(_) = $expr.concrete.as_ref() {
            true
        } else {
            false
        }
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::is!(@done $mod::[<Concrete $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::is!(@done [<Concrete $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_codegen::is!(@done $variant, $expr)
    };
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// mod foo {
///     #[arc_codegen::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = arc_codegen::enwrap!(foo::Bar::Baz, 5);
/// let y = arc_codegen::unwrap!(foo::Bar::Baz, x);
/// ```
#[macro_export]
macro_rules! unwrap {
    (@done $path:path, $expr:expr) => {
        if let $path(v) = $expr.concrete.as_ref() {
            v.clone()
        } else {
            unreachable!()
        }
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::unwrap!(@done $mod::[<Concrete $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_codegen::paste!(arc_codegen::unwrap!(@done [<Concrete $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_codegen::unwrap!(@done $variant, $expr)
    };
}
