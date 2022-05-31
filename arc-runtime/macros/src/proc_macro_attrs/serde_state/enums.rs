use proc_macro as pm;
use quote::quote;

pub(crate) fn serde_state(_: syn::AttributeArgs, mut item: syn::ItemEnum) -> pm::TokenStream {
    item.attrs.extend([
        syn::parse_quote!(#[derive(SerializeState, DeserializeState)]),
        syn::parse_quote!(#[serde(serialize_state = "SerdeState", deserialize_state = "SerdeState")])
    ]);

    item.variants.iter_mut().for_each(|variant| {
        variant
            .fields
            .iter_mut()
            .for_each(|field| field.attrs.push(syn::parse_quote!(#[serde(state)])))
    });

    quote!(#item).into()
}
