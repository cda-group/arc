use proc_macro::TokenStream;
use syn::TypeParam;

pub fn derive_send(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(unsafe impl #impl_generics Send for #name #type_generics #where_clause {}).into()
}

pub fn derive_sync(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(unsafe impl #impl_generics Sync for #name #type_generics #where_clause {}).into()
}

pub fn derive_unpin(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(impl #impl_generics Unpin for #name #type_generics #where_clause {}).into()
}

pub fn derive_deep_clone(mut input: syn::DeriveInput) -> TokenStream {
    for param in &mut input.generics.params {
        if let syn::GenericParam::Type(TypeParam { bounds, .. }) = param {
            bounds.push(syn::parse_quote!(DeepClone));
        }
    }

    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    match &input.data {
        syn::Data::Struct(data) => {
            let fields = data.fields.iter().enumerate().map(|(i, field)| {
                if let Some(name) = &field.ident {
                    quote::quote!(#name: <_ as DeepClone>::deep_clone(&self.#name))
                } else {
                    let index = syn::Index::from(i);
                    quote::quote!(#index: <_ as DeepClone>::deep_clone(&self.#index))
                }
            });
            quote::quote!(
                impl #impl_generics DeepClone for #name #type_generics #where_clause {
                    fn deep_clone(&self) -> Self {
                        Self {
                            #(#fields),*
                        }
                    }
                }
            )
            .into()
        }
        syn::Data::Enum(data) => {
            let variants = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                let names = variant.fields.iter().enumerate().map(|(i, field)| {
                    if let Some(name) = &field.ident {
                        quote::quote!(#name)
                    } else {
                        let index = syn::Index::from(i);
                        let id =
                            syn::Ident::new(&format!("field{}", i), proc_macro2::Span::call_site());
                        quote::quote!(#index: #id)
                    }
                });
                let fields = variant.fields.iter().enumerate().map(|(i, field)| {
                    if let Some(name) = &field.ident {
                        quote::quote!(#name: <_ as DeepClone>::deep_clone(&self.#name))
                    } else {
                        let index = syn::Index::from(i);
                        let id =
                            syn::Ident::new(&format!("field{}", i), proc_macro2::Span::call_site());
                        quote::quote!(#index: <_ as DeepClone>::deep_clone(&#id))
                    }
                });
                quote::quote!(Self::#name { #(#names),* } => Self::#name { #(#fields),* })
            });
            quote::quote!(
                impl #impl_generics DeepClone for #name #type_generics #where_clause {
                    fn deep_clone(&self) -> Self {
                        match self {
                            #(#variants),*
                        }
                    }
                }
            )
            .into()
        }
        syn::Data::Union(_) => unreachable!(),
    }
}
