use proc_macro as pm;

pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    let block = &item.block;
    let id = item.sig.ident;
    quote::quote! (
        pub fn #id() {
            #[rewrite]
            fn main() #block
            Manager::<Task>::start(main);
        }
    )
    .into()
}
