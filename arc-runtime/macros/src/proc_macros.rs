use crate::new_id;

use proc_macro as pm;
use proc_macro::TokenStream;
use proc_macro2 as pm2;

pub fn call(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            if args.len() == 1 && !args.trailing_punct() {
                quote::quote!(#func((#args,), ctx)).into()
            } else {
                quote::quote!(#func((#args), ctx)).into()
            }
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn spawn(input: syn::Expr) -> TokenStream {
    quote::quote!(spawn(Task { tag: #input.into() }, ctx)).into()
}

pub fn call_async(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            if args.len() == 1 && !args.trailing_punct() {
                quote::quote!(#func((#args,), ctx).await).into()
            } else {
                quote::quote!(#func((#args), ctx).await).into()
            }
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn call_indirect(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            if args.len() == 1 && !args.trailing_punct() {
                quote::quote!((#func.ptr)((#args,), ctx)).into()
            } else {
                quote::quote!((#func.ptr)((#args), ctx)).into()
            }
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn call_async_indirect(input: syn::Expr) -> TokenStream {
    match input {
        syn::Expr::Call(e) => {
            let func = e.func;
            let args = e.args;
            if args.len() == 1 && !args.trailing_punct() {
                quote::quote!((#func.ptr)((#args,), ctx).await).into()
            } else {
                quote::quote!((#func.ptr)((#args), ctx).await).into()
            }
        }
        _ => panic!("Expected function call expression"),
    }
}

pub fn enwrap(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut path: syn::Path = parse(&mut iter);
    let mut wrapper = path.clone();
    wrapper.segments.pop();
    wrapper.segments.push(syn::parse_quote!(new));
    inner_enum_path(&mut path);
    let data: syn::Expr = parse(&mut iter);
    quote::quote!(#wrapper(#path(#data), ctx)).into()
}

pub fn is(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut path: syn::Path = parse(&mut iter);
    inner_enum_path(&mut path);
    let data: syn::Expr = parse(&mut iter);
    quote::quote!(matches!(**#data, #path(_))).into()
}

pub fn unwrap(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut path: syn::Path = parse(&mut iter);
    inner_enum_path(&mut path);
    let expr: syn::Expr = parse(&mut iter);
    quote::quote!(if let #path(v) = **#expr { v } else { unreachable!() }).into()
}

pub fn new(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let mut data: syn::ExprStruct = parse(&mut iter);
    let mut wrapper = data.path.clone();
    wrapper.segments.push(syn::parse_quote!(new));
    inner_struct_path(&mut data.path);
    quote::quote!(#wrapper(#data, ctx)).into()
}

pub fn access(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: syn::Expr = parse(&mut iter);
    let ident: syn::Ident = parse(&mut iter);
    quote::quote!(#data.#ident).into()
}

pub fn val(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: syn::Expr = parse(&mut iter);
    quote::quote!(#data).into()
}

pub fn vector(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let data: Vec<syn::Expr> = parse_all(&mut iter);
    quote::quote!(Vector::from_vec(vec![#(#data),*], ctx)).into()
}

pub fn erase(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let expr: syn::Expr = parse(&mut iter);
    let ident: syn::Ident = parse(&mut iter);
    let (wrapper_impl, wrapper_cons) = generate_wrapper(&ident);
    let wrapper = wrapper_cons(expr);
    quote::quote!(
        {
            #wrapper_impl
            Erased::erase(#wrapper, ctx)
        }
    )
    .into()
}

pub fn unerase(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let expr: syn::Expr = parse(&mut iter);
    let id: syn::Ident = parse(&mut iter);
    let (wrapper_impl, _) = generate_wrapper(&id);
    quote::quote!(
        {
            #wrapper_impl
            Erased::unerase::<#id>(#expr, ctx)
        }
    )
    .into()
}

/// Transition to a new state.
pub fn transition(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let state: syn::Expr = parse(&mut iter);
    quote::quote!(return #state.into()).into()
}

fn generate_wrapper(id: &syn::Ident) -> (pm2::TokenStream, impl Fn(syn::Expr) -> pm2::TokenStream) {
    let span = id.span().unwrap().start();
    let line = span.line;
    let column = span.column;
    let id: syn::Ident = new_id(format!("Wrapper_{}_{}", line, column));
    let inner_id: syn::Ident = new_id(format!("InnerWrapper_{}_{}", line, column));
    let wrapper_impl = quote::quote!(
        use arc_runtime::prelude::*;
        #[derive(Clone, Debug, Send, Sync, Unpin, Trace, From, Deref)]
        #[repr(transparent)]
        pub struct #id(pub #inner_id);
        #[derive(Clone, Debug, Send, Sync, Unpin, Trace)]
        #[repr(transparent)]
        pub struct #inner_id(pub super::#id);

        impl CopyTo for #id {
            fn copy_to(&self, src: Heap, dst: Heap) -> Self::T {
                Self::erase(#id(#inner_id(self.0.0.copy_to(src, dst))), ctx)
            }
        }
    );
    let wrapper_cons = move |expr| quote::quote!(#id(#inner_id(#expr)));
    (wrapper_impl, wrapper_cons)
}

fn inner_enum_path(path: &mut syn::Path) {
    let mut x = path.segments.iter_mut();
    match (x.next(), x.next(), x.next()) {
        (Some(_), Some(i), Some(_)) => i.ident = new_id(format!("Inner{}", i.ident)),
        (Some(i), Some(_), None) => i.ident = new_id(format!("Inner{}", i.ident)),
        (Some(_), None, None) => {}
        _ => unreachable!(),
    }
}

fn inner_struct_path(path: &mut syn::Path) {
    let mut x = path.segments.iter_mut();
    match (x.next(), x.next()) {
        (Some(_), Some(i)) => i.ident = new_id(format!("Inner{}", i.ident)),
        (Some(i), None) => i.ident = new_id(format!("Inner{}", i.ident)),
        _ => unreachable!(),
    }
}

fn parse<T: syn::parse::Parse>(input: &mut impl Iterator<Item = pm::TokenTree>) -> T {
    let mut stream = pm::TokenStream::new();
    for token in input.by_ref() {
        match token {
            pm::TokenTree::Punct(t) if t.as_char() == ',' => break,
            _ => stream.extend([token]),
        }
    }
    syn::parse::<T>(stream).unwrap()
}

fn parse_all<T: syn::parse::Parse>(input: &mut impl Iterator<Item = pm::TokenTree>) -> Vec<T> {
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
