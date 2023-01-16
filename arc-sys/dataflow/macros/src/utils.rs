use proc_macro::TokenStream;
use proc_macro2 as pm2;
use proc_macro as pm;

pub(crate) fn new_id(s: impl ToString) -> syn::Ident {
    syn::Ident::new(&s.to_string(), pm2::Span::call_site())
}

pub(crate) fn get_metas(attr: &[syn::NestedMeta]) -> Vec<syn::Meta> {
    attr.iter()
        .filter_map(|a| match a {
            syn::NestedMeta::Meta(m) => Some(m.clone()),
            _ => None,
        })
        .collect()
}

pub(crate) fn _has_attr_key(name: &str, attr: &[syn::Attribute]) -> bool {
    attr.iter()
        .any(|a| matches!(a.parse_meta(), Ok(syn::Meta::Path(x)) if x.is_ident(name)))
}

pub(crate) fn has_meta_key(name: &str, meta: &[syn::Meta]) -> bool {
    meta.iter()
        .any(|m| matches!(m, syn::Meta::Path(x) if x.is_ident(name)))
}

pub(crate) fn has_meta_name_val(name: &str, meta: &[syn::Meta]) -> bool {
    meta.iter()
        .any(|m| matches!(m, syn::Meta::NameValue(x) if x.path.is_ident(name)))
}

pub(crate) fn _has_nested_meta_key(name: &str, meta: &[syn::NestedMeta]) -> bool {
    meta.iter()
        .any(|m| matches!(m, syn::NestedMeta::Meta(syn::Meta::Path(x)) if x.is_ident(name)))
}

#[allow(unused)]
pub(crate) fn get_attr_val<T: syn::parse::Parse>(name: &str, attr: &[syn::NestedMeta]) -> T {
    attr.iter()
        .find_map(|arg| match arg {
            syn::NestedMeta::Meta(meta) => match meta {
                syn::Meta::NameValue(nv) if nv.path.is_ident(name) => match &nv.lit {
                    syn::Lit::Str(x) => {
                        Some(x.parse().expect("Expected attr value to be an identifier"))
                    }
                    _ => None,
                },
                _ => None,
            },
            syn::NestedMeta::Lit(_) => None,
        })
        .unwrap_or_else(|| panic!("`{} = <id>` missing from identifiers", name))
}

pub(crate) fn _split_name_type(params: Vec<syn::FnArg>) -> (Vec<syn::Ident>, Vec<syn::Type>) {
    params
        .into_iter()
        .map(|p| match p {
            syn::FnArg::Receiver(_) => unreachable!(),
            syn::FnArg::Typed(p) => match *p.pat {
                syn::Pat::Ident(i) => (i.ident, *p.ty),
                _ => unreachable!(),
            },
        })
        .unzip()
}

pub(crate) struct ParsableNamedTypeField {
    pub(crate) field: syn::Field,
}

pub(crate) struct ParsableNamedExprField {
    pub(crate) field: syn::FieldValue,
}

impl syn::parse::Parse for ParsableNamedTypeField {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::parse::Result<Self> {
        let field = syn::Field::parse_named(input)?;
        Ok(ParsableNamedTypeField { field })
    }
}

impl syn::parse::Parse for ParsableNamedExprField {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::parse::Result<Self> {
        let field = syn::FieldValue::parse(input)?;
        Ok(ParsableNamedExprField { field })
    }
}

pub(crate) fn parse_type_fields(input: TokenStream) -> Vec<syn::Field> {
    let input = pm2::TokenStream::from(input);
    let punctuated_fields: syn::punctuated::Punctuated<ParsableNamedTypeField, syn::Token![,]> =
        syn::parse_quote!(#input);
    punctuated_fields.into_iter().map(|f| f.field).collect()
}

pub(crate) fn parse_expr_fields(input: TokenStream) -> Vec<syn::FieldValue> {
    let input = pm2::TokenStream::from(input);
    let punctuated_fields: syn::punctuated::Punctuated<ParsableNamedExprField, syn::Token![,]> =
        syn::parse_quote!(#input);
    punctuated_fields.into_iter().map(|f| f.field).collect()
}

pub(crate) fn parse<T: syn::parse::Parse>(input: &mut impl Iterator<Item = pm::TokenTree>) -> T {
    let mut stream = pm::TokenStream::new();
    for token in input.by_ref() {
        match token {
            pm::TokenTree::Punct(t) if t.as_char() == ',' => break,
            _ => stream.extend([token]),
        }
    }
    syn::parse::<T>(stream).unwrap()
}

pub(crate) fn parse_all<T: syn::parse::Parse>(input: &mut impl Iterator<Item = pm::TokenTree>) -> Vec<T> {
    let mut nodes = Vec::new();
    let mut stream = pm::TokenStream::new();
    for token in input.by_ref() {
        match token {
            pm::TokenTree::Punct(t) if t.as_char() == ',' => {
                nodes.push(syn::parse::<T>(stream).unwrap());
                stream = pm::TokenStream::new();
            }
            _ => stream.extend([token]),
        }
    }
    nodes
}

pub(crate) fn chars(id: &pm2::Ident) -> pm2::TokenStream {
    let s = id.to_string();
    let chars = s.chars().into_iter().map(|c| crate::utils::new_id(c));
    quote::quote!((#(chars::#chars,)*))
}

