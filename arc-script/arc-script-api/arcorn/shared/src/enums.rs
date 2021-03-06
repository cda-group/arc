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
/// use arc_script::arcorn;
/// mod foo {
///     use arc_script::arcorn;
///     #[arcorn::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
/// let x = arcorn::enwrap!(foo::Bar::Baz, 5);
/// ```
#[macro_export]
macro_rules! enwrap {
    (@done $path:path, $expr:expr) => {
        $path($expr).wrap()
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::enwrap!(@done $mod::[<Enum $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::enwrap!(@done [<Enum $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_script::arcorn::enwrap!(@done $variant, $expr)
    };
}

/// Returns `true` if enum is a certain variant, else `false`.
///
/// ```
/// use arc_script::arcorn;
/// mod foo {
///     use arc_script::arcorn;
///     #[arcorn::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = arcorn::enwrap!(foo::Bar::Baz, 5);
/// assert!(arcorn::is!(foo::Bar::Baz, x));
/// ```
#[macro_export]
macro_rules! is {
    (@done $path:path, $expr:expr) => {
        if let Some($path(_)) = &$expr.this {
            true
        } else {
            false
        }
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::is!(@done $mod::[<Enum $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::is!(@done [<Enum $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_script::arcorn::is!(@done $variant, $expr)
    };
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// use arc_script::arcorn;
/// mod foo {
///     use arc_script::arcorn;
///     #[arcorn::rewrite]
///     pub enum Bar {
///         Baz(i32),
///         Qux(i32)
///     }
/// }
///
/// let x = arcorn::enwrap!(foo::Bar::Baz, 5);
/// let y = arcorn::unwrap!(foo::Bar::Baz, x);
/// ```
#[macro_export]
macro_rules! unwrap {
    (@done $path:path, $expr:expr) => {
        if let Some($path(v)) = $expr.this {
            v
        } else {
            unreachable!()
        }
    };
    ($mod:ident :: $enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::unwrap!(@done $mod::[<Enum $enum>]::$variant, $expr))
    };
    ($enum:ident :: $variant:ident , $expr:expr) => {
        arc_script::arcorn::paste!(arc_script::arcorn::unwrap!(@done [<Enum $enum>]::$variant, $expr))
    };
    ($variant:ident , $expr:expr) => {
        arc_script::arcorn::unwrap!(@done $variant, $expr)
    };
}
