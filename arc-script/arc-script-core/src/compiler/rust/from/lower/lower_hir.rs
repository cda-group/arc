use crate::compiler::shared::Lower;

use crate::compiler::hir;
use crate::compiler::info::Info;
use crate::compiler::rust;

use super::Context;

use quote::quote;
use syn::parse_quote;

impl Lower<(), Context<'_>> for hir::HIR {
    fn lower(&self, ctx: &mut Context<'_>) {
        self.defs.values().for_each(|item| item.lower(ctx));
    }
}

impl Lower<(), Context<'_>> for hir::Item {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) {
        match &self.kind {
            hir::ItemKind::Alias(i)   => unreachable!(),
            hir::ItemKind::Enum(i)    => i.lower(ctx),
            hir::ItemKind::Fun(i)     => i.lower(ctx),
            hir::ItemKind::State(i)   => todo!(),
            hir::ItemKind::Task(i)    => i.lower(ctx),
            hir::ItemKind::Extern(i)  => {}
            hir::ItemKind::Variant(i) => {}
        }
    }
}

impl Lower<(), Context<'_>> for hir::Enum {
    fn lower(&self, ctx: &mut Context<'_>) -> () {
        let name = self.name.lower(ctx);
        let variants = self.variants.iter().map(|v| v.lower(ctx));
        let enum_item = syn::parse_quote! {
            pub enum #name {
                #(#variants),*
            }
        };
        ctx.rust.items.push(syn::Item::Enum(enum_item));
    }
}

impl Lower<(), Context<'_>> for hir::Fun {
    fn lower(&self, ctx: &mut Context<'_>) {
        let name = self.name.lower(ctx);
        let rtv = self.rtv.lower(ctx);
        let body = self.body.lower(ctx);
        let params = self.params.iter().map(|p| p.lower(ctx));
        let fn_item = syn::parse_quote! {
            pub fn #name(#(#params),*) -> #rtv {
                #body
            }
        };
        ctx.rust.items.push(syn::Item::Fn(fn_item));
    }
}

impl Lower<(), Context<'_>> for hir::Task {
    fn lower(&self, ctx: &mut Context<'_>) {
        let name = self.name.lower(ctx);
        let params = self.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>();
        let param_ids = self
            .params
            .iter()
            .map(|p| p.kind.lower(ctx))
            .collect::<Vec<_>>();
        let struct_item = syn::parse_quote! {
            #[derive(ArconState)]
            pub struct #name {
                #(#[ephemeral] pub #params),*
            }
        };
        let ity = self.iports.first().unwrap().lower(ctx);
        let oty = self.iports.first().unwrap().lower(ctx);
        let elem_id = self.on.param.kind.lower(ctx);
        let body = self.on.body.lower(ctx);
        let impl_item = syn::parse_quote! {
            use arcon::prelude::*;
            impl Operator for #name {
                type IN = #ity;
                type OUT = #oty;
                type TimerState = ArconNever;
                type OperatorState = Self;

                fn handle_element(
                    &mut self,
                    #elem_id: ArconElement<Self::IN>,
                    mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
                ) -> OperatorResult<()> {
                    let (#(#param_ids),*) = (#(self.#param_ids.clone()),*);
                    #body;
                    Ok(())
                }

                arcon::ignore_timeout!();
                arcon::ignore_persist!();
            }
        };
        ctx.rust.items.push(syn::Item::Struct(struct_item));
        ctx.rust.items.push(syn::Item::Impl(impl_item));
    }
}

impl Lower<syn::Ident, Context<'_>> for hir::Name {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Ident {
        let name = ctx.info.names.resolve(self.id);
        let span = proc_macro2::Span::call_site();
        syn::Ident::new(name, span)
    }
}

impl Lower<syn::Ident, Context<'_>> for hir::Path {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Ident {
        let path = ctx.info.resolve_to_names(self.id);
        let (_, path) = path.split_at(1);
        let name = path.join("_");
        let span = proc_macro2::Span::call_site();
        syn::Ident::new(&name, span)
    }
}

impl Lower<syn::FnArg, Context<'_>> for hir::Param {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::FnArg {
        let ty = self.tv.lower(ctx);
        let id = self.kind.lower(ctx);
        parse_quote!(#id: #ty)
    }
}

impl Lower<syn::Ident, Context<'_>> for hir::ParamKind {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Ident {
        match self {
            hir::ParamKind::Var(x) => {
                let x = x.lower(ctx);
                parse_quote!(#x)
            }
            hir::ParamKind::Ignore => parse_quote!(_),
            hir::ParamKind::Err => unreachable!(),
        }
    }
}

impl Lower<proc_macro2::TokenStream, Context<'_>> for hir::Expr {
    fn lower(&self, ctx: &mut Context<'_>) -> proc_macro2::TokenStream {
        match &self.kind {
            hir::ExprKind::Access(e, f) => {
                let e = e.lower(ctx);
                let f = f.lower(ctx);
                quote!(#e.#f)
            }
            hir::ExprKind::Array(es) => {
                let es = es.iter().map(|e| e.lower(ctx));
                quote!([#(#es),*])
            }
            hir::ExprKind::BinOp(e0, op, e1) => {
                let e0 = e0.lower(ctx);
                let op = op.lower(ctx);
                let e1 = e1.lower(ctx);
                quote!(#e0 #op #e1)
            }
            hir::ExprKind::Call(e, es) => {
                let e = e.lower(ctx);
                let es = es.iter().map(|e| e.lower(ctx));
                quote!(#e(#(#es),*))
            }
            hir::ExprKind::Emit(e) => {
                let e = e.lower(ctx);
                quote!(ctx.output(#e))
            }
            hir::ExprKind::If(e0, e1, e2) => {
                let e0 = e0.lower(ctx);
                let e1 = e1.lower(ctx);
                let e2 = e2.lower(ctx);
                quote!(if #e0 { #e1 } else { #e2 })
            }
            hir::ExprKind::Item(x) => {
                let x = x.lower(ctx);
                quote!(#x)
            }
            hir::ExprKind::Let(x, e0, e1) => {
                let x = x.lower(ctx);
                let e0 = e0.lower(ctx);
                let e1 = e1.lower(ctx);
                quote!(let #x = #e0; #e1)
            }
            hir::ExprKind::Lit(l) => {
                let l = l.lower(ctx);
                quote!(#l)
            }
            hir::ExprKind::Log(e) => {
                let e = e.lower(ctx);
                quote!(#e)
            }
            hir::ExprKind::Loop(e) => {
                let e = e.lower(ctx);
                quote!(loop { #e })
            }
            hir::ExprKind::Project(e, i) => {
                let e = e.lower(ctx);
                let i = syn::Index::from(i.id);
                quote!(#e.#i)
            }
            hir::ExprKind::Struct(efs) => todo!(),
            hir::ExprKind::Tuple(es) => {
                let es = es.iter().map(|e| e.lower(ctx));
                quote!((#(#es),*))
            }
            hir::ExprKind::UnOp(op, e) => {
                let op = op.lower(ctx);
                let e = e.lower(ctx);
                quote!(#op #e)
            }
            hir::ExprKind::Var(x) => {
                let x = x.lower(ctx);
                quote!(#x)
            }
            hir::ExprKind::Enwrap(x, e) => {
                let x = x.lower(ctx);
                let e = e.lower(ctx);
                quote!(#x(#e))
            }
            hir::ExprKind::Unwrap(x, e) => {
                let x = x.lower(ctx);
                let e = e.lower(ctx);
                quote!(arcorn::unwrap!(#e, #x))
            }
            hir::ExprKind::Is(x, e) => {
                let x = x.lower(ctx);
                let e = e.lower(ctx);
                quote!(arcorn::is!(#e, #x))
            }
            hir::ExprKind::Return(e) => quote!(return;),
            hir::ExprKind::Break => quote!(break;),
            hir::ExprKind::Err => unreachable!(),
        }
    }
}

impl Lower<syn::Type, Context<'_>> for hir::TypeId {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Type {
        ctx.info.types.resolve(*self).lower(ctx)
    }
}

impl Lower<syn::Type, Context<'_>> for hir::Type {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Type {
        match &self.kind {
            hir::TypeKind::Array(t, s) => todo!(),
            hir::TypeKind::Fun(ts, t) => {
                let t = t.lower(ctx);
                let ts = ts.iter().map(|t| t.lower(ctx));
                parse_quote!(fn(#(#ts),*) -> #t)
            }
            hir::TypeKind::Map(t0, t1) => todo!(),
            hir::TypeKind::Nominal(x) => {
                let x = x.lower(ctx);
                parse_quote!(#x)
            }
            hir::TypeKind::Optional(t) => {
                let t = t.lower(ctx);
                parse_quote!(Option<#t>)
            }
            hir::TypeKind::Scalar(s) => {
                let s = s.lower(ctx);
                parse_quote!(#s)
            }
            hir::TypeKind::Set(t) => todo!(),
            hir::TypeKind::Stream(t) => todo!(),
            hir::TypeKind::Struct(fts) => todo!(),
            hir::TypeKind::Task(ts0, ts1) => todo!(),
            hir::TypeKind::Tuple(ts) => {
                let ts = ts.iter().map(|t| t.lower(ctx));
                parse_quote!((#(#ts),*))
            }
            hir::TypeKind::Unknown => unreachable!(),
            hir::TypeKind::Vector(t) => todo!(),
            hir::TypeKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<syn::Type, Context<'_>> for hir::ScalarKind {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::Type {
        match self {
            hir::ScalarKind::Bool => parse_quote!(bool),
            hir::ScalarKind::Char => parse_quote!(char),
            hir::ScalarKind::F32  => parse_quote!(f32),
            hir::ScalarKind::F64  => parse_quote!(f64),
            hir::ScalarKind::I8   => parse_quote!(i8),
            hir::ScalarKind::I16  => parse_quote!(i16),
            hir::ScalarKind::I32  => parse_quote!(i32),
            hir::ScalarKind::I64  => parse_quote!(i64),
            hir::ScalarKind::U8   => parse_quote!(u8),
            hir::ScalarKind::U16  => parse_quote!(u16),
            hir::ScalarKind::U32  => parse_quote!(u32),
            hir::ScalarKind::U64  => parse_quote!(u64),
            hir::ScalarKind::Null => todo!(),
            hir::ScalarKind::Str  => todo!(),
            hir::ScalarKind::Unit => parse_quote!(()),
            hir::ScalarKind::Bot  => todo!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<syn::BinOp, Context<'_>> for hir::BinOp {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::BinOp {
        match self.kind {
            hir::BinOpKind::Add  => parse_quote!(+),
            hir::BinOpKind::And  => parse_quote!(+),
            hir::BinOpKind::Band => parse_quote!(&),
            hir::BinOpKind::Bor  => parse_quote!(|),
            hir::BinOpKind::Bxor => parse_quote!(^),
            hir::BinOpKind::Div  => parse_quote!(/),
            hir::BinOpKind::Equ  => parse_quote!(==),
            hir::BinOpKind::Geq  => parse_quote!(>=),
            hir::BinOpKind::Gt   => parse_quote!(>),
            hir::BinOpKind::Leq  => parse_quote!(<),
            hir::BinOpKind::Lt   => parse_quote!(<=),
            hir::BinOpKind::Mul  => parse_quote!(*),
            hir::BinOpKind::Mod  => parse_quote!(%),
            hir::BinOpKind::Neq  => parse_quote!(!=),
            hir::BinOpKind::Or   => parse_quote!(||),
            hir::BinOpKind::Pipe => todo!(),
            hir::BinOpKind::Pow  => todo!(),
            hir::BinOpKind::Seq  => todo!(),
            hir::BinOpKind::Sub  => parse_quote!(-),
            hir::BinOpKind::Xor  => todo!(),
            hir::BinOpKind::Err  => unreachable!(),
        }
    }
}

impl Lower<syn::UnOp, Context<'_>> for hir::UnOp {
    fn lower(&self, ctx: &mut Context<'_>) -> syn::UnOp {
        match self.kind {
            hir::UnOpKind::Neg => parse_quote!(-),
            hir::UnOpKind::Not => parse_quote!(!),
            hir::UnOpKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<proc_macro2::TokenStream, Context<'_>> for hir::LitKind {
    fn lower(&self, ctx: &mut Context<'_>) -> proc_macro2::TokenStream {
        match self {
            hir::LitKind::Bool(v) => quote!(#v),
            hir::LitKind::Char(v) => quote!(#v),
            hir::LitKind::F32(v)  => quote!(#v),
            hir::LitKind::F64(v)  => quote!(#v),
            hir::LitKind::I8(v)   => quote!(#v),
            hir::LitKind::I16(v)  => quote!(#v),
            hir::LitKind::I32(v)  => quote!(#v),
            hir::LitKind::I64(v)  => quote!(#v),
            hir::LitKind::U8(v)   => quote!(#v),
            hir::LitKind::U16(v)  => quote!(#v),
            hir::LitKind::U32(v)  => quote!(#v),
            hir::LitKind::U64(v)  => quote!(#v),
            hir::LitKind::Str(v)  => todo!(),
            hir::LitKind::Time(v) => todo!(),
            hir::LitKind::Unit    => quote!(()),
            hir::LitKind::Err     => unreachable!(),
        }
    }
}
