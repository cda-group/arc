use proc_macro::TokenStream;

pub fn call(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            quote::quote!(#func(#args)).into()
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn call_async(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            quote::quote!(#func(#args).await).into()
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn call_indirect(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            quote::quote!((#func.ptr)(#args)).into()
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn enwrap(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let path: syn::Path = crate::utils::parse(&mut iter);
    let data: syn::Expr = crate::utils::parse(&mut iter);
    quote::quote!(#path(#data)).into()
}

pub fn is(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let path: syn::Path = crate::utils::parse(&mut iter);
    let data: syn::Expr = crate::utils::parse(&mut iter);
    quote::quote!(matches!(#data, #path(_))).into()
}

pub fn unwrap(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let path: syn::Path = crate::utils::parse(&mut iter);
    let expr: syn::Expr = crate::utils::parse(&mut iter);
    quote::quote!(if let #path(v) = &#expr { v.clone() } else { unreachable!() }).into()
}

pub fn new(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut data: syn::ExprStruct = crate::utils::parse(&mut iter);
    let mut wrapper_path = data.path.clone();
    wrapper_path.segments.push(syn::parse_quote!(new));
    let mut iter = data.path.segments.iter_mut();
    match (iter.next(), iter.next()) {
        (Some(_), Some(seg)) => seg.ident = crate::utils::new_id(format!("_{}", seg.ident)),
        (Some(seg), None) => seg.ident = crate::utils::new_id(format!("_{}", seg.ident)),
        _ => unreachable!(),
    }
    quote::quote!(#wrapper_path(#data)).into()
}

pub fn access(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: syn::Expr = crate::utils::parse(&mut iter);
    let ident: syn::Ident = crate::utils::parse(&mut iter);
    quote::quote!(#data.#ident).into()
}

pub fn val(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: syn::Expr = crate::utils::parse(&mut iter);
    quote::quote!(#data.clone()).into()
}

pub fn vector(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: Vec<syn::Expr> = crate::utils::parse_all(&mut iter);
    quote::quote!(Vector::from_vec(vec![#(#data),*])).into()
}

pub fn get(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: syn::Expr = crate::utils::parse(&mut iter);
    let id: syn::Ident = crate::utils::parse(&mut iter);
    let chars = crate::utils::chars(&id);
    quote::quote!(<_ as Accessor<#chars, _>>::get(#data)).into()
}

pub fn label(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let id: syn::Ident = crate::utils::parse(&mut iter);
    let chars = crate::utils::chars(&id);
    quote::quote!(#chars).into()
}

#[allow(non_snake_case)]
pub fn Record(input: TokenStream) -> TokenStream {
    let fields = crate::utils::parse_type_fields(input);
    let (ids, tys): (Vec<_>, Vec<_>) = fields.into_iter().map(|f| (f.ident.unwrap(), f.ty)).unzip();
    let labels = ids.iter().map(crate::utils::chars);
    quote::quote!(impl #(Accessor<#labels, #tys>)+*+Copy).into()
}

pub fn record(input: TokenStream) -> TokenStream {
    let fields = crate::utils::parse_expr_fields(input);
    let (ids, exprs): (Vec<_>, Vec<_>) = fields
        .into_iter()
        .map(|f| {
            (
                match f.member {
                    syn::Member::Named(id) => id,
                    syn::Member::Unnamed(_) => panic!("Expected named fields"),
                },
                f.expr,
            )
        })
        .unzip();
    let (exprs, tys): (Vec<_>, Vec<_>) = exprs
        .into_iter()
        .map(|expr| match expr {
            syn::Expr::Cast(e) => (e.expr, e.ty),
            _ => panic!("Expected cast expression"),
        })
        .unzip();
    let line = std::line!();
    let id = crate::utils::new_id(format!("_{}", line));
    quote::quote!(
        {
            #[derive(Accessor, Copy, Clone)]
            struct #id { #(#ids: #tys,)* }
            #id { #(#ids: #exprs,)* }
        }
    )
    .into()
}

#[allow(non_snake_case)]
pub fn Enum(input: TokenStream) -> TokenStream {
    let fields = crate::utils::parse_type_fields(input);
    let (ids, tys): (Vec<_>, Vec<_>) = fields.into_iter().map(|f| (f.ident.unwrap(), f.ty)).unzip();
    let labels = ids.iter().map(crate::utils::chars);
    quote::quote!(impl #(Accessor<#labels, Option<#tys>>)+*+Copy).into()
}
