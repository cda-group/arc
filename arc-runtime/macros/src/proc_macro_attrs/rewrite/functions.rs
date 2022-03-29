use crate::has_attr_key;
use proc_macro as pm;
use proc_macro2 as pm2;
// use quote::quote;
use std::collections::HashMap;
use syn::visit_mut::VisitMut;

#[cfg(feature = "legacy")]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    quote::quote!(#item).into()
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    use crate::new_id;
    Visitor::default().visit_item_fn_mut(&mut item);
    let (ids, tys): (Vec<_>, Vec<_>) = item
        .sig
        .inputs
        .iter()
        .map(|x| match x {
            syn::FnArg::Receiver(_) => unreachable!(),
            syn::FnArg::Typed(t) => (&t.pat, &t.ty),
        })
        .unzip();

    let output = &item.sig.output;

    let id = new_id(format!("_{}", item.sig.ident));
    let wrapper_id = &item.sig.ident;
    let wrapper_item: syn::ItemFn = syn::parse_quote!(
        #[inline(always)]
        pub fn #wrapper_id((#(#ids,)*) : (#(#tys,)*), ctx: Context) #output {
            #id(#(#ids,)* ctx)
        }
    );
    item.sig.ident = id;
    item.sig.inputs.push(syn::parse_quote!(ctx: Context));
    quote::quote!(
        use arc_runtime::prelude::*;
        #wrapper_item #item
    ).into()
}

pub(crate) struct Visitor {
    scopes: Vec<HashMap<syn::Ident, MemKind>>,
}

impl Default for Visitor {
    fn default() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }
}

#[derive(Debug)]
enum MemKind {
    Heap,
    Stack,
}

impl VisitMut for Visitor {
    // Every function call must pass an implicit context parameter
    fn visit_expr_call_mut(&mut self, i: &mut syn::ExprCall) {
        i.args.push(syn::parse_quote!(ctx));
        syn::visit_mut::visit_expr_call_mut(self, i);
    }

    fn visit_pat_ident_mut(&mut self, i: &mut syn::PatIdent) {
        let kind = MemKind::Heap;

        self.scopes
            .last_mut()
            .map(|s| s.insert(i.ident.clone(), kind))
            .unwrap();
    }

    fn visit_pat_type_mut(&mut self, i: &mut syn::PatType) {
        let kind = if is_primitive(&i.ty) {
            MemKind::Stack
        } else {
            MemKind::Heap
        };

        self.scopes
            .last_mut()
            .map(|s| s.insert(get_pat_ident(&i.pat), kind))
            .unwrap();
    }

    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        self.scopes.push(HashMap::new());
        syn::visit_mut::visit_block_mut(self, i);
        self.scopes.pop();
    }

    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        syn::visit_mut::visit_expr_mut(self, i);
        if let syn::Expr::Path(expr) = i {
            if let Some(ident) = get_path_ident(&expr.path) {
                if let Some(MemKind::Heap) = self.scopes.iter().rev().find_map(|s| s.get(&ident)) {
                    *i = syn::parse_quote!(#ident.clone())
                }
            }
        }
    }

    fn visit_expr_macro_mut(&mut self, i: &mut syn::ExprMacro) {
        syn::visit_mut::visit_expr_macro_mut(self, i);
        i.mac.tokens = self.visit_token_stream(i.mac.tokens.clone());
    }

    // Visit expr before pattern
    fn visit_local_mut(&mut self, i: &mut syn::Local) {
        if let Some(it) = &mut i.init {
            self.visit_expr_mut(&mut *(it).1);
        }
        self.visit_pat_mut(&mut i.pat);
    }

    // Every let binding must use the context parameter for allocation
    fn visit_stmt_mut(&mut self, i: &mut syn::Stmt) {
        syn::visit_mut::visit_stmt_mut(self, i);
        if let syn::Stmt::Local(l) = i {
            if has_attr_key("alloc", &l.attrs) {
                let expr = &l.init.as_ref().unwrap().1;
                match &l.pat {
                    syn::Pat::Ident(pat) => {
                        *i = syn::parse_quote!(letroot!(#pat = ctx.mutator().shadow_stack(), #expr););
                    }
                    syn::Pat::Type(pat) => {
                        if !is_primitive(&*pat.ty) {
                            *i = syn::parse_quote!(letroot!(#pat = ctx.mutator().shadow_stack(), #expr););
                        }
                    }
                    _ => todo!(),
                }
            }
        }
    }
}

fn is_primitive(t: &syn::Type) -> bool {
    match t {
        syn::Type::Path(p) => [
            "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "unit",
        ]
        .contains(&p.path.segments.last().unwrap().ident.to_string().as_str()),
        syn::Type::Reference(tr) => match &*tr.elem {
            syn::Type::Path(p) => {
                ["str"].contains(&p.path.segments.last().unwrap().ident.to_string().as_str())
            }
            _ => false,
        },
        _ => false,
    }
}

fn get_pat_ident(p: &syn::Pat) -> syn::Ident {
    if let syn::Pat::Ident(i) = &*p {
        i.ident.clone()
    } else {
        panic!("Expected an identifier")
    }
}

fn get_path_ident(p: &syn::Path) -> Option<syn::Ident> {
    if p.segments.len() == 1 {
        Some(p.segments[0].ident.clone())
    } else {
        None
    }
}

impl Visitor {
    fn visit_token_stream(&mut self, tokens: pm2::TokenStream) -> pm2::TokenStream {
        let mut result = pm2::TokenStream::new();
        for token in tokens {
            let token = match &token {
                pm2::TokenTree::Group(g) => {
                    let stream = self.visit_token_stream(g.stream());
                    let delim = g.delimiter();
                    pm2::Group::new(delim, stream).into()
                }
                pm2::TokenTree::Punct(_) => token,
                pm2::TokenTree::Ident(i) => {
                    if let Some(MemKind::Heap) = self.scopes.iter().rev().find_map(|s| s.get(&i)) {
                        result.extend(quote::quote!((#i.clone())));
                        continue;
                    }
                    token
                }
                pm2::TokenTree::Literal(_) => token,
            };
            result.extend([token]);
        }
        result
    }
}
