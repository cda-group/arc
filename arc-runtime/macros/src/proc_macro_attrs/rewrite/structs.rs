use proc_macro as pm;
use quote::quote;

use crate::new_id;

#[cfg(not(feature = "legacy"))]
#[allow(unused)]
pub(crate) fn rewrite(args: syn::AttributeArgs, struct_item: syn::ItemStruct) -> pm::TokenStream {
    let abstract_id = struct_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", struct_item.ident));
    let sharable_mod_id = new_id(format!("sharable_struct_{}", struct_item.ident));
    let sendable_mod_id = new_id(format!("sendable_struct_{}", struct_item.ident));

    let mut concrete_sharable_struct_item = struct_item.clone();
    let mut concrete_sendable_struct_item = struct_item;

    concrete_sharable_struct_item.ident = concrete_id.clone();
    concrete_sendable_struct_item.ident = concrete_id.clone();

    // Generate the sendable struct
    concrete_sharable_struct_item
        .fields
        .iter_mut()
        .for_each(|f| {
            let ty = f.ty.clone();
            f.ty = syn::parse_quote!(super::#ty);
        });

    // Generate the sendable struct
    concrete_sendable_struct_item
        .fields
        .iter_mut()
        .for_each(|f| {
            let ty = f.ty.clone();
            f.ty = syn::parse_quote!(<super::#ty as DynSharable>::T);
        });

    let field_id = concrete_sendable_struct_item
        .fields
        .iter()
        .map(|f| &f.ident)
        .collect::<Vec<_>>();

    quote!(

        use arc_runtime::prelude::*;
        pub mod #sharable_mod_id {
            use arc_runtime::prelude::*;

            #[derive(Clone, Debug, Send, Sync, Alloc, Unpin, From, Deref, Abstract, Collectable, Finalize, Trace)]
            pub struct #abstract_id(pub Gc<#concrete_id>);

            #[derive(Clone, Debug, Collectable, Finalize, Trace)]
            #concrete_sharable_struct_item
        }

        mod #sendable_mod_id {
            use arc_runtime::prelude::*;

            #[derive(Clone, Debug, Deref, From, Abstract, Deserialize, Serialize)]
            #[from(forward)]
            pub struct #abstract_id(pub Box<#concrete_id>);

            #[derive(Clone, Debug, Deserialize, Serialize)]
            #concrete_sendable_struct_item
        }

        use #sharable_mod_id::#abstract_id;
        use #sharable_mod_id::#concrete_id;

        impl DynSharable for #sharable_mod_id::#abstract_id {
            type T = #sendable_mod_id::#abstract_id;
            fn into_sendable(&self, ctx: Context) -> Self::T {
                #sendable_mod_id::#concrete_id {
                    #(#field_id: (self.0).#field_id.clone().into_sendable(ctx)),*
                }.into()
            }
        }

        impl DynSendable for #sendable_mod_id::#abstract_id {
            type T = #sharable_mod_id::#abstract_id;
            fn into_sharable(&self, ctx: Context) -> Self::T {
                #sharable_mod_id::#concrete_id {
                    #(#field_id: (self.0).#field_id.into_sharable(ctx)),*
                }.alloc(ctx)
            }
        }

    )
    .into()
}

#[cfg(feature = "legacy")]
#[allow(unused)]
pub(crate) fn rewrite(_: syn::AttributeArgs, mut struct_item: syn::ItemStruct) -> pm::TokenStream {
    let abstract_id = struct_item.ident.clone();
    let concrete_id = new_id(format!("Concrete{}", struct_item.ident));
    let mod_id = new_id(format!("send_{}", struct_item.ident));

    struct_item.ident = concrete_id.clone();

    quote!(

        use arc_runtime::prelude::*;
        #[derive(Clone, Debug, From, Deref)]
        #[from(forward)]
        pub struct #abstract_id(pub std::rc::Rc<#concrete_id>);

        #[derive(Clone, Debug)]
        #struct_item
    )
    .into()
}
