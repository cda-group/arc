use proc_macro as pm;
use quote::quote;

use crate::new_id;

#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, struct_item: syn::ItemStruct) -> pm::TokenStream {
    let abstract_id = struct_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", struct_item.ident));
    let mod_id = new_id(format!("send_{}", struct_item.ident));

    let mut sharable_struct_item = struct_item.clone();
    let mut sendable_struct_item = struct_item;

    sharable_struct_item.ident = concrete_id.clone();
    sendable_struct_item.ident = concrete_id.clone();

    // Generate the sendable struct
    sendable_struct_item.fields.iter_mut().for_each(|f| {
        let ty = f.ty.clone();
        f.ty = syn::parse_quote!(<super::#ty as arc_script::codegen::Convert>::T);
    });

    let field_id = sendable_struct_item
        .fields
        .iter()
        .map(|f| &f.ident)
        .collect::<Vec<_>>();

    quote!(

        #[derive(Clone, Debug, arc_script::codegen::derive_more::Deref)]
        pub struct #abstract_id {
            pub concrete: std::rc::Rc<#concrete_id>,
        }

        impl arc_script::codegen::Convert for #concrete_id {
            type T = #abstract_id;
            fn convert(self) -> Self::T {
                Self::T {
                    concrete: std::rc::Rc::new(self),
                }
            }
        }

        #[derive(Clone, Debug)]
        #sharable_struct_item

        mod #mod_id {

            #[derive(Clone, Debug, arc_script::codegen::derive_more::Deref)]
            pub struct #abstract_id {
                pub concrete: Box<#concrete_id>,
            }

            // Concrete to Abstract
            impl arc_script::codegen::Convert for #concrete_id {
                type T = #abstract_id;
                fn convert(self) -> Self::T {
                    Self::T {
                        concrete: Box::new(self),
                    }
                }
            }

            #[derive(Clone, Debug)]
            #sendable_struct_item

            // Sharable to Sendable
            impl arc_script::codegen::Convert for super::#abstract_id {
                type T = #abstract_id;
                fn convert(self) -> Self::T {
                    #abstract_id {
                        concrete: Box::new(#concrete_id {
                            #(#field_id: self.concrete.as_ref().#field_id.clone().convert()),*
                        })
                    }
                }
            }

            // Sendable to Sharable
            impl arc_script::codegen::Convert for #abstract_id {
                type T = super::#abstract_id;
                fn convert(self) -> Self::T {
                    super::#abstract_id {
                        concrete: std::rc::Rc::new(super::#concrete_id {
                            #(#field_id: self.concrete.#field_id.convert()),*
                        })
                    }
                }
            }
        }
    )
    .into()
}
