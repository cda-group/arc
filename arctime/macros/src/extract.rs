#![allow(unused)]

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;
use syn::parse::*;
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::*;

pub(crate) struct TaskComponents {
    pub(crate) mod_name: syn::Ident,
    pub(crate) task_name: syn::Ident,
    pub(crate) iport_enum: syn::ItemEnum,
    pub(crate) oport_enum: syn::ItemEnum,
    pub(crate) iport_enum_name: syn::Ident,
    pub(crate) oport_enum_name: syn::Ident,
    pub(crate) iport_name: Vec<syn::Ident>,
    pub(crate) iport_type: Vec<syn::Type>,
    pub(crate) oport_name: Vec<syn::Ident>,
    pub(crate) oport_type: Vec<syn::Type>,
    pub(crate) state_name: Vec<syn::Ident>,
    pub(crate) state_type: Vec<syn::Type>,
    pub(crate) handler_name: syn::Ident,
}

pub(crate) fn extract(attr: AttributeArgs, module: ItemMod) -> TaskComponents {
    let mod_name = module.ident.clone();

    let items = module.content.expect("Expected module to contain items").1;

    let task = items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) => Some(item),
            _ => None,
        })
        .expect("Expected a task-struct in module");

    let task_name = task.ident.clone();
    let state_name = task
        .fields
        .iter()
        .map(|f| f.ident.clone().expect("Expected field identifier"))
        .collect();
    let state_type = task.fields.iter().map(|f| f.ty.clone()).collect();

    let mut enums = items.iter().filter_map(|item| match item {
        Item::Enum(item) => Some(item),
        _ => None,
    });
    let iport_enum = enums.next().expect("Expected input port enum").clone();
    let oport_enum = enums.next().expect("Expected output port enum").clone();

    let iport_enum_name = iport_enum.ident.clone();
    let oport_enum_name = oport_enum.ident.clone();

    let handler: syn::Ident = attr
        .iter()
        .find_map(|arg| match arg {
            NestedMeta::Meta(meta) => match meta {
                Meta::NameValue(nv) if nv.path.is_ident("handler") => match &nv.lit {
                    Lit::Str(x) => Some(
                        x.parse()
                            .expect("Expected handler value to be an identifier"),
                    ),
                    _ => None,
                },
                _ => None,
            },
            NestedMeta::Lit(lit) => None,
        })
        .expect(r#"`handler = <id>` missing from identifiers"#);

    let iport_name = iport_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect();
    let oport_name = oport_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect();

    let iport_type = iport_enum
        .variants
        .iter()
        .map(|variant| variant.fields.iter().next().unwrap().ty.clone())
        .collect();

    let oport_type = oport_enum
        .variants
        .iter()
        .map(|variant| variant.fields.iter().next().unwrap().ty.clone())
        .collect();

    TaskComponents {
        mod_name,
        task_name,
        iport_enum,
        oport_enum,
        iport_enum_name,
        oport_enum_name,
        iport_name,
        iport_type,
        oport_name,
        oport_type,
        state_name,
        state_type,
        handler_name: handler,
    }
}
