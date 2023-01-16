use proc_macro::TokenStream;

pub fn derive_notrace(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(
        impl #impl_generics Trace for #name #type_generics #where_clause {
            fn trace(&self, heap: Heap) { }
            fn root(&self, heap: Heap) { }
            fn unroot(&self, heap: Heap) { }
            fn copy(&self, heap: Heap) -> Self { *self }
        }
    )
    .into()
}

pub fn derive_trace(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    match &input.data {
        syn::Data::Struct(data) => {
            let field = data
                .fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    field
                        .ident
                        .as_ref()
                        .map(|ident| quote::quote!(#ident))
                        .unwrap_or_else(|| {
                            let index = syn::Index::from(index);
                            quote::quote!(#index)
                        })
                })
                .collect::<Vec<_>>();
            let copy = match &data.fields {
                syn::Fields::Named(_) => {
                    quote::quote!(Self {#(#field: self.#field.copy(heap)),*} )
                }
                syn::Fields::Unnamed(_) => {
                    quote::quote!(Self(#(self.#field.copy(heap)),*))
                }
                syn::Fields::Unit => {
                    quote::quote!(Self)
                }
            };
            quote::quote!(
                impl #impl_generics Trace for #name #type_generics #where_clause {
                    fn trace(&self, heap: Heap) {
                        #(self.#field.trace(heap));*
                    }
                    fn root(&self, heap: Heap) {
                        #(self.#field.root(heap));*
                    }
                    fn unroot(&self, heap: Heap) {
                        #(self.#field.unroot(heap));*
                    }
                    fn copy(&self, heap: Heap) -> Self {
                        #copy
                    }
                }
            )
            .into()
        }
        syn::Data::Enum(data) => {
            let variant = data.variants.iter().map(|v| &v.ident).collect::<Vec<_>>();
            let field = data
                .variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .enumerate()
                        .map(|(i, _)| crate::utils::new_id(format!("_{}", i)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let pattern = field
                .iter()
                .map(|f| {
                    if f.is_empty() {
                        quote::quote!()
                    } else {
                        quote::quote!((#(#f),*))
                    }
                })
                .collect::<Vec<_>>();
            let copy = field
                .iter()
                .map(|f| {
                    if f.is_empty() {
                        quote::quote!()
                    } else {
                        quote::quote!((#(#f.copy(heap)),*))
                    }
                })
                .collect::<Vec<_>>();
            quote::quote!(
                impl #impl_generics Trace for #name #type_generics #where_clause {
                    fn trace(&self, heap: Heap) {
                        match self {
                            #(Self::#variant #pattern => { #(#field.trace(heap));* },)*
                        }
                    }
                    fn root(&self, heap: Heap) {
                        match self {
                            #(Self::#variant #pattern => {#(#field.trace(heap));*},)*
                        }
                    }
                    fn unroot(&self, heap: Heap) {
                        match self {
                            #(Self::#variant #pattern => {#(#field.trace(heap));*},)*
                        }
                    }
                    fn copy(&self, heap: Heap) -> Self {
                        match self {
                            #(Self::#variant #pattern => Self::#variant #copy),*
                        }
                    }
                }
            )
            .into()
        }
        syn::Data::Union(_) => unreachable!(),
    }
}

pub fn derive_nodebug(input: syn::DeriveInput) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(
        impl #impl_generics std::fmt::Debug for #name #type_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", stringify!(#name))
            }
        }
    )
    .into()
}

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

pub fn derive_noserde(input: syn::DeriveInput) -> TokenStream {
    let mut deserialize_generics = input.generics.clone();
    deserialize_generics.params.push(syn::parse_quote!('de));
    let (serialize_impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let (deserialize_impl_generics, _, _) = deserialize_generics.split_for_impl();
    let name = &input.ident;
    quote::quote!(
        impl #serialize_impl_generics Serialize for #name #type_generics #where_clause {
            fn serialize<S: Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                panic!("Attempted to serialize an unserializable type {}", stringify!(#name))
            }
        }

        impl #deserialize_impl_generics Deserialize<'de> for #name #type_generics #where_clause {
            fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
                panic!("Attempted to deserialize an undeserializable type {}", stringify!(#name))
            }
        }
    )
    .into()
}

pub fn derive_accessor(input: syn::DeriveInput) -> TokenStream {
    let mut impls = Vec::new();
    match input.data {
        syn::Data::Struct(data) => {
            let name = &input.ident;
            let mut ids = Vec::new();
            let mut labels = Vec::new();
            let mut types = Vec::new();
            for data in data.fields.iter() {
                let t = &data.ty;
                let id = &data.ident;
                let chars = crate::utils::chars(id.as_ref().unwrap());
                ids.push(id.clone());
                labels.push(chars.clone());
                types.push(t.clone());
                let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
                impls.push(quote::quote!(
                    impl #impl_generics Accessor<#chars, #t> for #name #type_generics #where_clause {
                        fn get(self) -> #t {
                            self.#id
                        }
                    }
                ));
            }
            impls.push(quote::quote!(
                impl<T> Transform<T> for #name where T: #(Accessor<#labels, #types>+)* Copy + Clone {
                    fn transform(other: T) -> Self {
                        Self {
                            #(#ids: get!(other, #ids),)*
                        }
                    }
                }
            ));
        }
        syn::Data::Enum(data) => {
            let name = &input.ident;
            let mut ids = Vec::new();
            let mut labels = Vec::new();
            let mut types = Vec::new();
            for data in data.variants.iter() {
                let variant_id = &data.ident;
                let data = data.fields.iter().next().unwrap();
                let t = &data.ty;
                let id = &data.ident;
                let chars = crate::utils::chars(id.as_ref().unwrap());
                ids.push(id.clone());
                labels.push(chars.clone());
                types.push(t.clone());
                let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
                impls.push(quote::quote!(
                    impl #impl_generics Accessor<#chars, Option<#t>> for #name #type_generics #where_clause {
                        fn get(self) -> #t {
                            if let #name::#variant_id(data) = self.#id {
                                Some(data)
                            } else {
                                None
                            }
                        }
                    }
                ));
            }
            impls.push(quote::quote!(
                impl<T> Transform<T> for #name where T: #(Accessor<#labels, #types>+)* Copy + Clone {
                    fn transform(other: T) -> Self {
                        Self {
                            #(#ids: get!(other, #ids),)*
                        }
                    }
                }
            ));
        }
        syn::Data::Union(_) => todo!(),
    };
    quote::quote!(#(#impls)*).into()
}
