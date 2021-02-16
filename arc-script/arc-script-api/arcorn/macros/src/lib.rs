use proc_macro::TokenStream;

mod conv;

#[proc_macro]
pub fn conv(input: TokenStream) -> TokenStream {
    conv::execute(input)
}
