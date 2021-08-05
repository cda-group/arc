//! Codegen for enums

use proc_macro as pm;
use quote::quote;

use crate::new_id;

#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut enum_item: syn::ItemEnum) -> pm::TokenStream {
    let abstract_id = enum_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", abstract_id));
    let mod_id = new_id(format!("send_{}", abstract_id));

    let mut sharable_enum_item = enum_item.clone();
    let mut sendable_enum_item = enum_item;

    sharable_enum_item.ident = concrete_id.clone();
    sendable_enum_item.ident = concrete_id.clone();

    // Generate the sendable enum
    sendable_enum_item.variants.iter_mut().for_each(|v| {
        v.fields.iter_mut().for_each(|f| {
            let ty = f.ty.clone();
            f.ty = syn::parse_quote!(<super::#ty as arc_script::codegen::Convert>::T);
        })
    });

    let variant_id = sharable_enum_item
        .variants
        .iter()
        .map(|v| &v.ident)
        .collect::<Vec<_>>();

    quote!(

        #[derive(Clone, Debug)]
        pub struct #abstract_id {
            pub concrete: std::rc::Rc<#concrete_id>,
        }

        impl Convert for #concrete_id {
            type T = #abstract_id;
            fn convert(self) -> Self::T {
                Self::T {
                    concrete: std::rc::Rc::new(self),
                }
            }
        }

        #[derive(Clone, Debug)]
        #sharable_enum_item

        use #concrete_id::*;

        pub mod #mod_id {
            use arc_script::codegen::*;

            #[derive(Clone, Debug)]
            pub struct #abstract_id {
                pub concrete: Box<#concrete_id>,
            }

            impl Convert for #concrete_id {
                type T = #abstract_id;
                fn convert(self) -> Self::T {
                    Self::T {
                        concrete: Box::new(self),
                    }
                }
            }

            #[derive(Clone, Debug)]
            #sendable_enum_item

            // Sharable to Sendable
            impl Convert for super::#abstract_id {
                type T = #abstract_id;
                fn convert(self) -> Self::T {
                    match self.concrete.as_ref() {
                        #(
                            super::#concrete_id::#variant_id(x) => #abstract_id {
                                concrete: Box::new(#concrete_id::#variant_id(x.clone().convert()))
                            }
                        ),*
                    }
                }
            }

            // Sendable to Sharable
            impl Convert for #abstract_id {
                type T = super::#abstract_id;
                fn convert(self) -> Self::T {
                    match *self.concrete {
                        #(
                            #concrete_id::#variant_id(x) => super::#abstract_id {
                                concrete: std::rc::Rc::new(super::#concrete_id::#variant_id(x.convert()))
                            }
                        ),*
                    }
                }
            }

        }

    )
    .into()
}
