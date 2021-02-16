//! Macros for using arbitrary enums without having to
//! implement a bunch of boilerplate methods.
//!
//! The assumptions are:
//! * Each enum variant has exactly one field.
//! * Each enum is wrapped inside a struct (prost requirement).
//! * Each enum implements a method `.wrap()` to wrap it inside the struct.
//! * Each enum variant identifier is unique and in the global namspace.
//!
//! All of these assumptions will be ensured by the Arcorn interface.

/// Enwraps a value into an enum-variant.
///
/// ```
/// use arcorn::enwrap;
/// use derive_more::From;
/// struct Foo {
///     this: FooEnum
/// }
/// impl FooEnum {
///     fn wrap(self) -> Foo {
///         Foo {
///             this: self
///         }
///     }
/// }
/// enum FooEnum {
///     FooBar(i32),
///     FooBaz(i32)
/// }
/// use FooEnum::*;
///
/// let foo = enwrap!(FooBar, 5);
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
/// use arcorn::enwrap;
/// use arcorn::is;
/// use derive_more::From;
/// #[derive(From)]
/// struct Foo {
///     this: FooEnum
/// }
/// impl FooEnum {
///     fn wrap(self) -> Foo {
///         Foo {
///             this: self
///         }
///     }
/// }
/// enum FooEnum {
///     FooBar(i32),
///     FooBaz(i32)
/// }
/// use FooEnum::*;
///
/// let foo = enwrap!(FooBar, 5);
/// assert!(is!(FooBar, foo));
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
/// use arcorn::enwrap;
/// use arcorn::unwrap;
/// use derive_more::From;
/// #[derive(From)]
/// struct Foo {
///     this: FooEnum
/// }
/// impl FooEnum {
///     fn wrap(self) -> Foo {
///         Foo {
///             this: self
///         }
///     }
/// }
/// enum FooEnum {
///     FooBar(i32),
///     FooBaz(i32)
/// }
/// use FooEnum::*;
///
/// let foo = enwrap!(FooBar, 5);
/// let bar = unwrap!(FooBar, foo);
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
