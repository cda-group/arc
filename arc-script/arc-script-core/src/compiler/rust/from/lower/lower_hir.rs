use arc_script_core_shared::Lower;

use crate::compiler::hir;

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
            hir::ItemKind::Alias(_i)   => unreachable!(),
            hir::ItemKind::Enum(i)    => i.lower(ctx),
            hir::ItemKind::Fun(i)     => i.lower(ctx),
            hir::ItemKind::State(_i)   => todo!(),
            hir::ItemKind::Task(i)    => i.lower(ctx),
            hir::ItemKind::Extern(_i)  => {}
            hir::ItemKind::Variant(_i) => {}
        }
    }
}

impl Lower<(), Context<'_>> for hir::Enum {
    fn lower(&self, ctx: &mut Context<'_>) {
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
        let cons_name = self.name.lower(ctx);
        let task_name = syn::Ident::new(
            &format!("Task{}", cons_name),
            proc_macro2::Span::call_site(),
        );
        let params = self.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>();
        let param_ids = self
            .params
            .iter()
            .map(|p| p.kind.lower(ctx))
            .collect::<Vec<_>>();
        let clone_params = if param_ids.is_empty() {
            None
        } else {
            Some(quote!(let (#(#param_ids),*) = (#(self.#param_ids.clone()),*);))
        };
        let struct_item = syn::parse_quote! {
            #[derive(ArconState)]
            pub struct #task_name {
                #(#[ephemeral] pub #params),*
            }
        };
        let ity = if let hir::HubKind::Single(ity) = &self.ihub.kind {
            ity.lower(ctx)
        } else {
            todo!();
        };
        let oty = if let hir::HubKind::Single(oty) = &self.ohub.kind {
            oty.lower(ctx)
        } else {
            todo!();
        };
        let elem_id = self.on.param.kind.lower(ctx);
        let body = self.on.body.lower(ctx);
        let impl_item = syn::parse_quote! {
            impl Operator for #task_name {
                type IN  = #ity;
                type OUT = #oty;
                type TimerState = ArconNever;
                type OperatorState = Self;

                fn handle_element(
                    &mut self,
                    #elem_id: ArconElement<Self::IN>,
                    mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
                ) -> OperatorResult<()> {
                    #clone_params
                    #body;
                    Ok(())
                }

                arcon::ignore_timeout!();
                arcon::ignore_persist!();
            }
        };

        let fn_item = syn::parse_quote! {
            fn #cons_name(#(#params),*) -> OperatorBuilder<#task_name> {
                OperatorBuilder {
                    constructor: Arc::new(|b| #task_name { #(#params),* }),
                    conf: Default::default(),
                }
            }
        };
        ctx.rust.items.push(syn::Item::Struct(struct_item));
        ctx.rust.items.push(syn::Item::Impl(impl_item));
        ctx.rust.items.push(syn::Item::Fn(fn_item));
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
            Self::Var(x) => {
                let x = x.lower(ctx);
                parse_quote!(#x)
            }
            Self::Ignore => parse_quote!(_),
            Self::Err => unreachable!(),
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
                let t = ctx.info.types.resolve(e.tv);
                let e = e.lower(ctx);
                let es = es.iter().map(|e| e.lower(ctx)).collect::<Vec<_>>();
                if let hir::TypeKind::Fun(_, rtv) = t.kind {
                    if let hir::TypeKind::Stream(_) = ctx.info.types.resolve(rtv).kind {
                        let stream = es.get(0).unwrap();
                        let task = e;
                        quote!(Stream::operator(#stream, #task))
                    } else {
                        quote!(#e(#(#es),*))
                    }
                } else {
                    unreachable!()
                }
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
                quote!(println!("{:?}", #e))
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
            hir::ExprKind::Struct(_efs) => todo!(),
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
            hir::ExprKind::Return(_e) => quote!(return;),
            hir::ExprKind::Break => quote!(break;),
            hir::ExprKind::Todo => quote!(todo!()),
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
            hir::TypeKind::Array(_t, _s) => todo!(),
            hir::TypeKind::Fun(ts, t) => {
                let t = t.lower(ctx);
                let ts = ts.iter().map(|t| t.lower(ctx));
                parse_quote!(fn(#(#ts),*) -> #t)
            }
            hir::TypeKind::Map(_t0, _t1) => todo!(),
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
            hir::TypeKind::Set(_t) => todo!(),
            hir::TypeKind::Stream(t) => {
                let t = t.lower(ctx);
                parse_quote!(Stream<#t>)
            }
            hir::TypeKind::Struct(_fts) => todo!(),
            hir::TypeKind::Tuple(ts) => {
                let ts = ts.iter().map(|t| t.lower(ctx));
                parse_quote!((#(#ts),*))
            }
            hir::TypeKind::Unknown => unreachable!(),
            hir::TypeKind::Vector(_t) => todo!(),
            hir::TypeKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<syn::Type, Context<'_>> for hir::ScalarKind {
    fn lower(&self, _ctx: &mut Context<'_>) -> syn::Type {
        match self {
            Self::Bool => parse_quote!(bool),
            Self::Char => parse_quote!(char),
            Self::Bf16 => parse_quote!(bf16),
            Self::F16  => parse_quote!(f16),
            Self::F32  => parse_quote!(f32),
            Self::F64  => parse_quote!(f64),
            Self::I8   => parse_quote!(i8),
            Self::I16  => parse_quote!(i16),
            Self::I32  => parse_quote!(i32),
            Self::I64  => parse_quote!(i64),
            Self::U8   => parse_quote!(u8),
            Self::U16  => parse_quote!(u16),
            Self::U32  => parse_quote!(u32),
            Self::U64  => parse_quote!(u64),
            Self::Null => todo!(),
            Self::Str  => todo!(),
            Self::Unit => parse_quote!(()),
            Self::Bot  => todo!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<syn::BinOp, Context<'_>> for hir::BinOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> syn::BinOp {
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
            hir::BinOpKind::Mut  => parse_quote!(=),
            hir::BinOpKind::Err  => unreachable!(),
        }
    }
}

impl Lower<syn::UnOp, Context<'_>> for hir::UnOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> syn::UnOp {
        match self.kind {
            hir::UnOpKind::Neg => parse_quote!(-),
            hir::UnOpKind::Not => parse_quote!(!),
            hir::UnOpKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<proc_macro2::TokenStream, Context<'_>> for hir::LitKind {
    fn lower(&self, _ctx: &mut Context<'_>) -> proc_macro2::TokenStream {
        match self {
            Self::Bool(v) => quote!(#v),
            Self::Char(v) => quote!(#v),
            Self::Bf16(v) => {
                let v = v.to_f32();
                if v.is_nan() {
                    quote!(half::bf16::NAN)
                } else if v.is_infinite() {
                    quote!(half::bf16::INFINITE)
                } else {
                    quote!(half::bf16::from_f32(#v))
                }
            },
            Self::F16(v) => {
                let v = v.to_f32();
                if v.is_nan() {
                    quote!(half::f16::NAN)
                } else if v.is_infinite() {
                    quote!(half::f16::INFINITE)
                } else {
                    quote!(half::f16::from_f32(#v))
                }
            },
            Self::F32(v) => {
                if v.is_nan() {
                    quote!(f32::NAN)
                } else if v.is_infinite() {
                    quote!(f32::INFINITE)
                } else {
                    quote!(#v)
                }
            },
            Self::F64(v) => {
                if v.is_nan() {
                    quote!(f64::NAN)
                } else if v.is_infinite() {
                    quote!(f64::INFINITE)
                } else {
                    quote!(#v)
                }
            },
            Self::I8(v)   => quote!(#v),
            Self::I16(v)  => quote!(#v),
            Self::I32(v)  => quote!(#v),
            Self::I64(v)  => quote!(#v),
            Self::U8(v)   => quote!(#v),
            Self::U16(v)  => quote!(#v),
            Self::U32(v)  => quote!(#v),
            Self::U64(v)  => quote!(#v),
            Self::Str(_v)  => todo!(),
            Self::Time(_v) => todo!(),
            Self::Unit    => quote!(()),
            Self::Err     => unreachable!(),
        }
    }
}
