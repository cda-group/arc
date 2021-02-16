use indexmap::IndexMap as HashMap;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

/// Takes a bunch of structs and enums and generates boilerplate code.
pub(super) fn execute(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::File);

    let mut codegen = Codegen::new(input);

    codegen.build_arcon_enums();
    codegen.build_arcon_structs();

    TokenStream::from(codegen.generate())
}

struct Codegen {
    arcon: Arcon,
    arc: Arc,
    impls: Vec<syn::ItemImpl>,
}

#[rustfmt::skip]
impl Codegen {
    fn new(input: syn::File) -> Self {
        let mut structs = HashMap::new();
        let mut enums = HashMap::new();

        for item in input.items {
            match item {
                syn::Item::Enum(data) => {
                    enums.insert(data.ident.clone(), data);
                }
                syn::Item::Struct(data) => {
                    structs.insert(data.ident.clone(), data);
                }
                _ => panic!("Expected Enum or Struct found other item"),
            }
        }
        Self {
            arcon: Arcon {
                tag_counter: 0,
                enums: HashMap::new(),
                structs: HashMap::new(),
            },
            arc: Arc { structs, enums },
            impls: Vec::new(),
        }
    }

    /// Assigns unique tags to each enum variant and struct field.
    fn build_arcon_enums(&mut self) {
        let enums = &self.arc.enums;
        let structs = &self.arc.structs;
        let arcon = &mut self.arcon;
        let impls = &mut self.impls;
        for (enum_id, mut item) in enums.clone().into_iter() {
            item.attrs.push(syn::parse_quote!(#[derive(Oneof, Clone)]));
            let range = arcon.tag_range(item.variants.len());
            let mut arcon2arc = Vec::new();
            let mut arc2arcon = Vec::new();
            item.variants
                .iter_mut()
                .zip(range.clone().into_iter())
                .for_each(|(variant, tag)| {
                    let variant_id = &variant.ident;
                    let mut field = variant.fields.iter_mut().next().unwrap();
                    let ty = ArcType::parse(&field.ty, structs, enums);
                    let tag = tag.to_string();
                    // Generate enum
                    match &ty {
                        ArcType::RcStruct(id) => field.ty = syn::parse_quote!(Box<#id>),
                        ArcType::RcString => field.ty = syn::parse_quote!(String),
                        _ => {}
                    }
                    let attr = match &ty {
                        ArcType::RcStruct(_) => syn::parse_quote!(#[prost(message, tag = #tag)]),
                        ArcType::RcString    => syn::parse_quote!(#[prost(string, tag = #tag)]),
                        ArcType::I32         => syn::parse_quote!(#[prost(int32, tag = #tag)]),
                        ArcType::I64         => syn::parse_quote!(#[prost(int64, tag = #tag)]),
                        ArcType::U32         => syn::parse_quote!(#[prost(uint32, tag = #tag)]),
                        ArcType::U64         => syn::parse_quote!(#[prost(uint64, tag = #tag)]),
                        ArcType::RcEnum(_)   => panic!("Enum variant cannot have Enum type"),
                    };
                    variant.attrs.push(attr);
                    // Generate to/from conversions
                    match &ty {
                        ArcType::RcStruct(_) => {
                            arc2arcon.push(quote!(super::#enum_id::#variant_id(x) => #enum_id::#variant_id(Box::new(x.as_ref().into()))));
                            arcon2arc.push(quote!(#enum_id::#variant_id(x) => super::#enum_id::#variant_id(Rc::new((*x).into()))));
                        }
                        ArcType::I32 | ArcType::I64 | ArcType::U32 | ArcType::U64 => {
                            arc2arcon.push(quote!(super::#enum_id::#variant_id(x) => #enum_id::#variant_id(x.clone())));
                            arcon2arc.push(quote!(#enum_id::#variant_id(x) => super::#enum_id::#variant_id(x.clone())));
                        }
                        ArcType::RcString => {
                            arc2arcon.push(quote!(super::#enum_id::#variant_id(x) => #enum_id::#variant_id(x.as_ref().clone())));
                            arcon2arc.push(quote!(#enum_id::#variant_id(x) => super::#enum_id::#variant_id(Rc::new(x))));
                        }
                        ArcType::RcEnum(_) => panic!("Enum variant cannot have Enum type"),
                    }
                });
            impls.push(syn::parse_quote! {
                impl From<&'_ super::#enum_id> for #enum_id {
                    fn from(data: &super::#enum_id) -> #enum_id {
                        match data { #(#arc2arcon),* }
                    }
                }
            });
            impls.push(syn::parse_quote! {
                impl From<#enum_id> for super::#enum_id {
                    fn from(data: #enum_id) -> super::#enum_id {
                        match data { #(#arcon2arc),* }
                    }
                }
            });
            arcon
                .enums
                .insert(enum_id.clone(), ArconEnum { tags: range, item });
        }
    }

    fn build_arcon_structs(&mut self) {
        let enums = &self.arc.enums;
        let structs = &self.arc.structs;
        let arcon = &mut self.arcon;
        let impls = &mut self.impls;
        for (struct_id, mut item) in structs.clone().into_iter() {
            item.attrs.push(syn::parse_quote!(#[derive(Arcon, Clone, Message)]));
            // TODO: These numbers should not be hardcoded.
            item.attrs.push(syn::parse_quote!(#[arcon(reliable_ser_id = 13, version = 1)]));
            let mut arc2arcon = Vec::new();
            let mut arcon2arc = Vec::new();
            item.fields.iter_mut().for_each(|field| {
                let ty = ArcType::parse(&field.ty, structs, enums);
                let field_id = &field.ident.as_ref().unwrap();
                // Generate struct
                let attr = if let ArcType::RcEnum(id) = &ty {
                    let tags = arcon.enums.get(id).unwrap().tags.clone();
                    let tags = tags.into_iter().map(|tag| tag.to_string()).collect::<Vec<_>>().join(", ");
                    let string_id = id.to_string();
                    field.ty = syn::parse_quote!(Option<#id>);
                    syn::parse_quote!(#[prost(oneof = #string_id, tags = #tags)])
                } else {
                    let tag = arcon.next_tag().to_string();
                    match &ty {
                        ArcType::RcStruct(id) => field.ty = syn::parse_quote!(#id),
                        ArcType::RcString => field.ty = syn::parse_quote!(String),
                        _ => {}
                    }
                    match &ty {
                        ArcType::RcStruct(_) => syn::parse_quote!(#[prost(message, tag = #tag)]),
                        ArcType::I32         => syn::parse_quote!(#[prost(int32, tag = #tag)]),
                        ArcType::I64         => syn::parse_quote!(#[prost(int64, tag = #tag)]),
                        ArcType::U32         => syn::parse_quote!(#[prost(uint32, tag = #tag)]),
                        ArcType::U64         => syn::parse_quote!(#[prost(uint64, tag = #tag)]),
                        ArcType::RcString    => syn::parse_quote!(#[prost(string, tag = #tag)]),
                        ArcType::RcEnum(_)   => unreachable!()
                    }
                };
                field.attrs.push(attr);
                // Generate conversions to/from struct
                match &ty {
                    ArcType::RcStruct(_) | ArcType::RcString => {
                        arc2arcon.push(quote!(#field_id: data.#field_id.as_ref().into()));
                        arcon2arc.push(quote!(#field_id: Rc::new(data.#field_id.into())));
                    }
                    ArcType::RcEnum(_) => {
                        arc2arcon.push(quote!(#field_id: Some(data.#field_id.as_ref().into())));
                        arcon2arc.push(quote!(#field_id: Rc::new(data.#field_id.unwrap().into())));
                    }
                    ArcType::I32 | ArcType::I64 | ArcType::U32 | ArcType::U64 => {
                        arc2arcon.push(quote!(#field_id: data.#field_id.clone()));
                        arcon2arc.push(quote!(#field_id: data.#field_id.clone()));
                    }
                }
            });
            impls.push(syn::parse_quote! {
                impl From<&'_ super::#struct_id> for #struct_id {
                    fn from(data: &super::#struct_id) -> #struct_id {
                        #struct_id { #(#arc2arcon),* }
                    }
                }
            });
            impls.push(syn::parse_quote! {
                impl From<#struct_id> for super::#struct_id {
                    fn from(data: #struct_id) -> super::#struct_id {
                        super::#struct_id { #(#arcon2arc),* }
                    }
                }
            });
            arcon.structs.insert(struct_id.clone(), ArconStruct { item });
        }
    }

    fn generate(&self) -> proc_macro2::TokenStream {
        let arc_structs = self.arc.structs.values();
        let arc_enums = self.arc.enums.values();
        let arcon_structs = self.arcon.structs.values().map(|data| &data.item);
        let arcon_enums = self.arcon.enums.values().map(|data| &data.item);
        let impls = self.impls.iter();
        quote! {
            #(#arc_structs)*
            #(#arc_enums)*
            pub mod arcon_types {
                use prost::{Message, Oneof};
                use arcon_macros::Arcon;
                use std::rc::Rc;
                #(#arcon_structs)*
                #(#arcon_enums)*
                #(#impls)*
            }
        }
    }
}

type Tag = usize;

use std::ops::Range;

struct Arcon {
    tag_counter: Tag,
    enums: HashMap<syn::Ident, ArconEnum>,
    structs: HashMap<syn::Ident, ArconStruct>,
}

struct ArconEnum {
    tags: Range<Tag>,
    item: syn::ItemEnum,
}

struct ArconStruct {
    item: syn::ItemStruct,
}

impl Arcon {
    fn tag_range(&mut self, len: usize) -> Range<Tag> {
        let range = self.tag_counter..self.tag_counter + len;
        self.tag_counter += len;
        range
    }
    fn next_tag(&mut self) -> Tag {
        let tag = self.tag_counter;
        self.tag_counter += 1;
        tag
    }
}

struct Arc {
    structs: HashMap<syn::Ident, syn::ItemStruct>,
    enums: HashMap<syn::Ident, syn::ItemEnum>,
}

enum ArcType {
    RcStruct(syn::Ident),
    RcEnum(syn::Ident),
    RcString,
    I32,
    I64,
    U32,
    U64,
}

impl ArcType {
    fn parse(
        ty: &syn::Type,
        structs: &HashMap<syn::Ident, syn::ItemStruct>,
        enums: &HashMap<syn::Ident, syn::ItemEnum>,
    ) -> Self {
        if let syn::Type::Path(ty) = ty {
            let mut segments = ty.path.segments.iter();
            let segment = segments.next().unwrap();
            let id = segment.ident.clone();
            let args = &segment.arguments;
            assert!(segments.next().is_none(), "Found path of multiple segments");
            match id.to_string().as_str() {
                "Rc" => {
                    if let syn::PathArguments::AngleBracketed(args) = args {
                        let mut args = args.args.iter();
                        let arg = args.next().unwrap();
                        if let syn::GenericArgument::Type(ty) = arg {
                            if let syn::Type::Path(ty) = ty {
                                let mut segments = ty.path.segments.iter();
                                let segment = segments.next().unwrap();
                                assert!(
                                    segments.next().is_none(),
                                    "Found path of multiple segments"
                                );
                                let id = segment.ident.clone();
                                match id.to_string().as_str() {
                                    "String" => ArcType::RcString,
                                    _ if structs.contains_key(&id) => ArcType::RcStruct(id),
                                    _ if enums.contains_key(&id) => ArcType::RcEnum(id),
                                    _ => panic!("Rc referenced undefined Nominal-type"),
                                }
                            } else {
                                panic!("Rc expected Nominal-type, found something else")
                            }
                        } else {
                            panic!("Rc has unexpected parameter")
                        }
                    } else {
                        panic!("Rc has no args");
                    }
                }
                "i32" => ArcType::I32,
                "i64" => ArcType::I64,
                "u32" => ArcType::U32,
                "u64" => ArcType::U64,
                _ => panic!("Expected Arc-type, found something else"),
            }
        } else {
            panic!("Expected Nominal-type, found something else");
        }
    }
}
