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
/// #[arcorn::rewrite]
/// enum Foo {
///     FooBar(i32),
///     FooBaz(i32)
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
/// use arc_script::arcorn;
/// #[arcorn::rewrite]
/// enum Foo {
///     FooBar(i32),
///     FooBaz(i32)
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
        if let Some($variant(_)) = &$expr.this {
            true
        } else {
            false
        }
    }
}

/// Unwraps a value out of an enum-variant.
///
/// ```
/// use arc_script::arcorn;
/// #[arcorn::rewrite]
/// enum FooEnum {
///     FooBar(i32),
///     FooBaz(i32)
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
        if let Some($variant(v)) = $expr.this {
            v
        } else {
            unreachable!()
        }
    }
}
