extern crate proc_macro;

mod loc;
mod id;

use proc_macro::TokenStream;

/// Expands:
///
/// #[derive(Spanned)]
/// struct Foo { loc: Loc, bar: Bar, baz: Baz }
///
/// Into:
///
/// impl From<Spanned<(Bar, Baz)>> for Foo {
///     fn from(Spanned(file, lhs, (bar, baz), rhs): Spanned<(Bar, Baz)>) -> Self {
///         Self { bar, baz, loc: Loc::new(file, lhs..rhs) }
///     }
/// }
///
/// impl Foo {
///     fn syn(bar: Bar, baz: Baz) -> Self {
///         Self { bar, baz, loc: None }
///     }
/// }
#[proc_macro_derive(Loc)]
pub fn derive_loc(input: TokenStream) -> TokenStream {
    loc::derive(input)
}

#[proc_macro_derive(GetId)]
pub fn derive_id(input: TokenStream) -> TokenStream {
    id::derive(input)
}
