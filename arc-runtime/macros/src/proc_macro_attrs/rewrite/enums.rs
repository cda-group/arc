//! Codegen for enums

use proc_macro as pm;
use quote::quote;

use crate::new_id;

#[cfg(not(feature = "legacy"))]
#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut enum_item: syn::ItemEnum) -> pm::TokenStream {
    let abstract_id = enum_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", abstract_id));
    let sharable_mod_id = new_id(format!("sharable_enum_{}", abstract_id));
    let sendable_mod_id = new_id(format!("sendable_enum_{}", abstract_id));

    let mut concrete_sharable_enum_item = enum_item.clone();
    let mut concrete_sendable_enum_item = enum_item;

    // ADD VISITOR TO REWRITE Erased to Box<dyn Send>

    concrete_sharable_enum_item.ident = concrete_id.clone();
    concrete_sendable_enum_item.ident = concrete_id.clone();

    concrete_sharable_enum_item
        .variants
        .iter_mut()
        .for_each(|v| {
            v.fields.iter_mut().for_each(|f| {
                let ty = f.ty.clone();
                f.ty = syn::parse_quote!(super::#ty);
            })
        });

    concrete_sendable_enum_item
        .variants
        .iter_mut()
        .for_each(|v| {
            v.fields.iter_mut().for_each(|f| {
                let ty = f.ty.clone();
                f.ty = syn::parse_quote!(<super::#ty as DynSharable>::T);
            })
        });

    let variant_id = concrete_sharable_enum_item
        .variants
        .iter()
        .map(|v| &v.ident)
        .collect::<Vec<_>>();

    quote!(

        use arc_runtime::prelude::*;
        pub mod #sharable_mod_id {
            use arc_runtime::prelude::*;

            #[derive(Clone, Debug, Abstract, Send, Sync, Unpin, Alloc, Collectable, Finalize, Trace)]
            pub struct #abstract_id(pub Gc<#concrete_id>);

            #[derive(Clone, Debug, Collectable, Finalize, Trace)]
            #concrete_sharable_enum_item
        }

        pub mod #sendable_mod_id {
            use arc_runtime::prelude::*;
  
            #[derive(Clone, Debug, From, Abstract, Serialize, Deserialize)]
            #[from(forward)]
            pub struct #abstract_id(pub Box<#concrete_id>);
 
            #[derive(Clone, Debug, Serialize, Deserialize)]
            #concrete_sendable_enum_item
        }

        use #sharable_mod_id::#abstract_id;
        use #sharable_mod_id::#concrete_id::*;

        impl DynSharable for #sharable_mod_id::#abstract_id {
            type T = #sendable_mod_id::#abstract_id;
            fn into_sendable(&self, ctx: Context) -> Self::T {
                match &*self.0 {
                    #(
                        #sharable_mod_id::#concrete_id::#variant_id(x) =>
                        #sendable_mod_id::#concrete_id::#variant_id(x.clone().into_sendable(ctx)).into()
                    ),*
                }
            }
        }

        impl DynSendable for #sendable_mod_id::#abstract_id {
            type T = #sharable_mod_id::#abstract_id;
            fn into_sharable(&self, ctx: Context) -> Self::T {
                match &*self.0 {
                    #(
                        #sendable_mod_id::#concrete_id::#variant_id(x) =>
                        #sharable_mod_id::#concrete_id::#variant_id(x.into_sharable(ctx)).alloc(ctx)
                    ),*
                }
            }
        }

    )
    .into()
}

#[cfg(feature = "legacy")]
#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut enum_item: syn::ItemEnum) -> pm::TokenStream {
    let abstract_id = enum_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", abstract_id));
    let mod_id = new_id(format!("send_{}", abstract_id));

    enum_item.ident = concrete_id.clone();

    quote!(

        #[derive(Clone, Debug, From)]
        #[from(forward)]
        pub struct #abstract_id(pub std::rc::Rc<#concrete_id>);

        #[derive(Clone, Debug)]
        #enum_item

        use #concrete_id::*;
    )
    .into()
}
