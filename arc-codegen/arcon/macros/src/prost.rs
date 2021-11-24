use proc_macro2 as pm2;

pub(crate) fn ty_to_prost_attr(ty: &syn::Type, tag: Option<usize>) -> syn::Attribute {
    let ty = match &ty {
        syn::Type::Path(ty) => {
            let seg = ty.path.segments.iter().next().unwrap();
            match seg.ident.to_string().as_str() {
                "i32" => "int32",
                "i64" => "int64",
                "bool" => "bool",
                "f32" => "float",
                "f64" => "double",
                "u32" => "uint32",
                "u64" => "uint64",
                "String" => "string",
                // This case covers messages which are wrapped in Box<T> as well
                _ => "message",
            }
            .to_string()
        }
        syn::Type::Tuple(ty) if ty.elems.is_empty() => "message".to_string(),
        _ => panic!("#[codegen::rewrite] expects all types to be mangled and de-aliased."),
    };
    let ident = syn::Ident::new(&ty, pm2::Span::call_site());
    if let Some(tag) = tag {
        let lit = syn::LitStr::new(&format!("{}", tag), pm2::Span::call_site());
        syn::parse_quote!(#[prost(#ident, tag = #lit)])
    } else {
        syn::parse_quote!(#[prost(#ident, required)])
    }
}
