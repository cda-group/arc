use arc_script_core_shared::get;
use arc_script_core_shared::Bool;
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;

use crate::compiler::arcon::from::lower::lowerings::structs;
use crate::compiler::hir;
use crate::compiler::info::Info;

use super::Context;

use proc_macro2 as pm2;
use proc_macro2::TokenStream as Tokens;
use quote::quote;

impl Lower<Tokens, Context<'_>> for hir::HIR {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let defs = self
            .items
            .iter()
            .map(|item| self.defs.get(item).unwrap().lower(ctx))
            .collect::<Vec<_>>();
        let mangled_defs = ctx.mangled_defs.values();
        quote! {
            use arc_script::arcorn;
            #(#defs)*
            #(#mangled_defs)*
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Item {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        match &self.kind {
            hir::ItemKind::Alias(_i)  => unreachable!(),
            hir::ItemKind::Enum(i)    => i.lower(ctx),
            hir::ItemKind::Fun(i)     => i.lower(ctx),
            hir::ItemKind::State(_i)  => todo!(),
            hir::ItemKind::Task(i)    => i.lower(ctx),
            hir::ItemKind::Extern(_i) => quote!(),
            hir::ItemKind::Variant(i) => i.lower(ctx),
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Enum {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let name = self.path.lower(ctx);
        let variants = self
            .variants
            .iter()
            .map(|v| ctx.hir.defs.get(v).unwrap().lower(ctx));
        quote! {
            #[arcorn::rewrite]
            pub enum #name {
                #(#variants),*
            }
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Variant {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let name = self.path.lower(ctx);
        let ty = self.tv.lower(ctx);
        quote! {
            #name(#ty)
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Fun {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let name = self.path.lower(ctx);
        let rtv = self.rtv.lower(ctx);
        let body = self.body.lower(ctx);
        let params = self.params.iter().map(|p| p.lower(ctx));
        quote! {
            pub fn #name(#(#params),*) -> #rtv {
                #body
            }
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Task {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let cons_name = self.path.lower(ctx);
        let task_name = syn::Ident::new(&format!("Task{}", cons_name), pm2::Span::call_site());
        let handler_name =
            syn::Ident::new(&format!("Handler{}", cons_name), pm2::Span::call_site());

        let param_decls = self.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>();
        let param_ids = self
            .params
            .iter()
            .map(|p| p.kind.lower(ctx))
            .collect::<Vec<_>>();
        let (state_decls, state_inits): (Vec<_>, Vec<_>) =
            self.items.iter().filter_map(|x| x.lower_state(ctx)).unzip();
        let methods = self
            .items
            .iter()
            .filter_map(|item| item.lower_method(ctx))
            .collect::<Vec<_>>();
        let x = ctx.info.names.intern("value").into();

        let itv = get!(&self.ihub.kind, hir::HubKind::Single(x));
        let itv = get!(ctx.info.types.resolve(*itv).kind, hir::TypeKind::Stream(x));
        let fs = vec![(x, itv)].into_iter().collect();

        let ity = itv.lower(ctx);
        let struct_ity = ctx.info.types.intern(hir::TypeKind::Struct(fs)).lower(ctx);

        let otv = get!(&self.ohub.kind, hir::HubKind::Single(x));
        let otv = get!(ctx.info.types.resolve(*otv).kind, hir::TypeKind::Stream(x));
        let fs = vec![(x, otv)].into_iter().collect();

        let oty = otv.lower(ctx);
        let struct_oty = ctx.info.types.intern(hir::TypeKind::Struct(fs)).lower(ctx);

        let on = get!(&self.on, Some(on));
        let event = on.param.kind.lower(ctx);
        let body = on.body.lower(ctx);

        quote! {

            pub struct #handler_name<
                'i,
                'source,
                'timer,
                'channel,
                O: 'static + Operator,
                B: Backend,
                C: ComponentDefinition,
            > {
                pub op: &'i mut O,
                pub ctx: &'i mut OperatorContext<'source, 'timer, 'channel, O, B, C>,
                pub timestamp: Option<u64>,
            }

            #[derive(ArconState)]
            pub struct #task_name {
                #(#state_decls),*
                #(#[ephemeral] pub #param_decls),*
            }

            fn #cons_name(#(#param_decls),*) -> OperatorBuilder<#task_name> {
                OperatorBuilder {
                    constructor: Arc::new(move |b| #task_name {
                        #(#state_inits),*
                        #(#param_ids),*
                    }),
                    conf: Default::default(),
                }
            }

            impl Operator for #task_name {
                type IN  = #struct_ity;
                type OUT = #struct_oty;
                type TimerState = ArconNever;
                type OperatorState = Self;

                fn handle_element(
                    &mut self,
                    elem: ArconElement<Self::IN>,
                    ref mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
                ) -> OperatorResult<()> {
                    let ArconElement { timestamp, data } = elem;
                    let event = data.value;
                    let mut handler = #handler_name {
                        op: self,
                        ctx,
                        timestamp
                    };
                    handler.handle(event);
                    Ok(())
                }

                arcon::ignore_timeout!();
                arcon::ignore_persist!();
            }

            impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
                #handler_name<'i, 'source, 'timer, 'channel, #task_name, B, C>
            {
                fn handle(&mut self, #event: #ity) -> OperatorResult<()> {
                    #body;
                    Ok(())
                }

                fn emit(&mut self, #event: #oty) {
                    let data = #struct_oty { value: #event };
                    let elem = ArconElement { data, timestamp: self.timestamp };
                    self.ctx.output(elem);
                }

                #(#methods)*
            }

        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Name {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let name = ctx.info.names.resolve(self.id);
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(name, span);
        quote!(#ident)
    }
}

impl Lower<Tokens, Context<'_>> for hir::Path {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let path = ctx.info.resolve_to_names(self.id);
        let (_, path) = path.split_at(1);
        let name = path.join("_");
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(&name, span);
        quote!(#ident)
    }
}

impl Lower<Tokens, Context<'_>> for hir::Param {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let ty = self.tv.lower(ctx);
        let id = self.kind.lower(ctx);
        quote!(#id: #ty)
    }
}

impl Lower<Tokens, Context<'_>> for hir::ParamKind {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        match self {
            Self::Var(x) => x.lower(ctx),
            Self::Ignore => quote!(_),
            Self::Err => unreachable!(),
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::Expr {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let mut env = Map::default();
        self.block(ctx, &mut env, 0)
    }
}

fn new_id(pos: usize, depth: usize) -> Tokens {
    let id = syn::Ident::new(&format!("y_{}_{}", depth, pos), pm2::Span::call_site());
    quote!(#id)
}

impl hir::Expr {
    fn block(
        &self,
        ctx: &mut Context<'_>,
        env: &mut Map<syn::Ident, syn::Ident>,
        depth: usize,
    ) -> Tokens {
        let depth = depth + 1;
        let mut ops = Vec::new();
        let id = self.ssa(ctx, env, &mut ops, depth);
        ops.iter().enumerate().rev().fold(id, |acc, (i, x)| {
            let id = new_id(i, depth);
            quote!(let #id = #x; #acc)
        })
    }

    fn ssa(
        &self,
        ctx: &mut Context<'_>,
        env: &mut Map<syn::Ident, syn::Ident>,
        ops: &mut Vec<Tokens>,
        depth: usize,
    ) -> Tokens {
        let expr = match &self.kind {
            hir::ExprKind::Let(x, e0, e1) => {
                if let hir::ParamKind::Var(x) = x.kind {
                    let x0 = e0.ssa(ctx, env, ops, depth);
                    let x0 = syn::parse_quote!(#x0);
                    let x = x.lower(ctx);
                    let x = syn::parse_quote!(#x);
                    env.insert(x, x0);
                }
                return e1.ssa(ctx, env, ops, depth);
            }
            hir::ExprKind::Var(x, kind) => {
                let x = x.lower(ctx);
                match kind {
                    hir::VarKind::Local => {
                        let mut x = syn::parse_quote!(#x);
                        while let Some(next) = env.get(&x) {
                            x = next.clone();
                        }
                        return quote!(#x);
                    }
                    hir::VarKind::Member => quote!((self.op.#x)),
                }
            }
            hir::ExprKind::Access(e, f) => {
                let e = e.ssa(ctx, env, ops, depth);
                let f = f.lower(ctx);
                quote!(#e.#f)
            }
            hir::ExprKind::Array(es) => {
                let es = es.iter().map(|e| e.ssa(ctx, env, ops, depth));
                quote!([#(#es),*])
            }
            hir::ExprKind::BinOp(e0, op, e1) => {
                let tv = e1.tv;
                let e0 = e0.ssa(ctx, env, ops, depth);
                let e1 = e1.ssa(ctx, env, ops, depth);
                if let hir::BinOpKind::Pow = op.kind {
                    match tv {
                        _ if tv.is_float(ctx.info) => quote!(#e0.powf(#e1)),
                        _ if tv.is_int(ctx.info) => quote!(#e0.powi(#e1)),
                        _ => unreachable!(),
                    }
                } else {
                    let op = op.lower(ctx);
                    quote!(#e0 #op #e1)
                }
            }
            hir::ExprKind::Call(e, es) => {
                let t = ctx.info.types.resolve(e.tv);
                let e = e.ssa(ctx, env, ops, depth);
                let es = es
                    .iter()
                    .map(|e| e.ssa(ctx, env, ops, depth))
                    .collect::<Vec<_>>();
                let (_, rtv) = get!(t.kind, hir::TypeKind::Fun(tvs, rtv));
                if let hir::TypeKind::Stream(_) = ctx.info.types.resolve(rtv).kind {
                    let stream = es.get(0).unwrap();
                    let task = e;
                    quote!(Stream::operator(#stream, #task))
                } else {
                    quote!(#e(#(#es),*))
                }
            }
            hir::ExprKind::Emit(e) => {
                let e = e.ssa(ctx, env, ops, depth);
                quote!(self.emit(#e))
            }
            hir::ExprKind::If(e0, e1, e2) => {
                let e0 = e0.ssa(ctx, env, ops, depth);
                let e1 = e1.block(ctx, env, depth);
                let e2 = e2.block(ctx, env, depth);
                quote!(if #e0 { #e1 } else { #e2 })
            }
            // NOTE: Don't assign items to variables
            hir::ExprKind::Item(x) => match &ctx.hir.defs.get(x).unwrap().kind {
                hir::ItemKind::Fun(item) if matches!(item.kind, hir::FunKind::Method) => {
                    let x = x.lower(ctx);
                    return quote!(self.#x);
                }
                _ => return x.lower(ctx),
            },
            hir::ExprKind::Lit(l) => l.lower(ctx),
            hir::ExprKind::Log(e) => {
                let e = e.ssa(ctx, env, ops, depth);
                quote!(println!("{:?}", #e))
            }
            hir::ExprKind::Loop(e) => {
                let e = e.ssa(ctx, env, ops, depth);
                quote!(loop { #e })
            }
            hir::ExprKind::Project(e, i) => {
                let e = e.ssa(ctx, env, ops, depth);
                let i = syn::Index::from(i.id);
                quote!(#e.#i)
            }
            hir::ExprKind::Struct(efs) => {
                // NOTE: Lower here to ensure that the type definition is interned
                self.tv.lower(ctx);
                let ident = structs::mangle(self.tv, ctx);
                let efs = efs
                    .iter()
                    .map(|(x, e)| {
                        let x = x.lower(ctx);
                        let e = e.ssa(ctx, env, ops, depth);
                        quote!(#x: #e)
                    })
                    .collect::<Vec<_>>();
                quote!(#ident { #(#efs),* })
            }
            hir::ExprKind::Tuple(es) => {
                let es = es.iter().map(|e| e.ssa(ctx, env, ops, depth));
                quote!((#(#es),*))
            }
            hir::ExprKind::UnOp(op, e) => {
                let e = e.ssa(ctx, env, ops, depth);
                if let hir::UnOpKind::Boxed = &op.kind {
                    quote!(Box::new(#e))
                } else {
                    let op = op.lower(ctx);
                    quote!(#op #e)
                }
            }
            hir::ExprKind::Enwrap(x, e) => {
                let x = x.lower(ctx);
                let e = e.ssa(ctx, env, ops, depth);
                quote!(arcorn::enwrap!(#x, #e))
            }
            hir::ExprKind::Unwrap(x, e) => {
                let x = x.lower(ctx);
                let e = e.ssa(ctx, env, ops, depth);
                quote!(arcorn::unwrap!(#x, #e))
            }
            hir::ExprKind::Is(x, e) => {
                let x = x.lower(ctx);
                let e = e.ssa(ctx, env, ops, depth);
                quote!(arcorn::is!(#x, #e))
            }
            hir::ExprKind::Return(e) => {
                let e = e.ssa(ctx, env, ops, depth);
                quote!({return #e;})
            }
            hir::ExprKind::Break => quote!({
                break;
            }),
            hir::ExprKind::Todo => quote!(todo!()),
            hir::ExprKind::Err => unreachable!(),
        };
        let id = new_id(ops.len(), depth);
        ops.push(expr);
        quote!(#id)
    }
}

impl Lower<Tokens, Context<'_>> for hir::TypeId {
    fn lower(&self, ctx: &mut Context<'_>) -> Tokens {
        let ty = ctx.info.types.resolve(*self);
        match &ty.kind {
            hir::TypeKind::Array(_t, _s) => todo!(),
            hir::TypeKind::Fun(ts, t) => {
                let t = t.lower(ctx);
                let ts = ts.iter().map(|t| t.lower(ctx));
                quote!(fn(#(#ts),*) -> #t)
            }
            hir::TypeKind::Map(t0, t1) => {
                let t0 = t0.lower(ctx);
                let t1 = t1.lower(ctx);
                quote!(Map<#t0, #t1>)
            }
            hir::TypeKind::Nominal(x) => {
                let x = x.lower(ctx);
                quote!(#x)
            }
            hir::TypeKind::Optional(t) => {
                let t = t.lower(ctx);
                quote!(Option<#t>)
            }
            hir::TypeKind::Scalar(s) => {
                let s = s.lower(ctx);
                quote!(#s)
            }
            hir::TypeKind::Set(t) => {
                let t = t.lower(ctx);
                quote!(Set<T>)
            }
            hir::TypeKind::Stream(t) => {
                let x = ctx.info.names.intern("value").into();
                let fs = vec![(x, *t)].into_iter().collect();
                let t = ctx.info.types.intern(hir::TypeKind::Struct(fs)).lower(ctx);
                quote!(Stream<#t>)
            }
            hir::TypeKind::Struct(fts) => {
                let ident = structs::mangle(*self, ctx);
                if !ctx.mangled_defs.contains_key(&ident) {
                    let fts = fts
                        .iter()
                        .map(|(f, t)| {
                            let f = f.lower(ctx);
                            let t = t.lower(ctx);
                            quote!(#f : #t)
                        })
                        .collect::<Vec<_>>();
                    let derive_copy = self
                        .is_copyable((ctx.info as &Info, ctx.hir))
                        .as_option()
                        .map(|_| quote!(#[derive(Copy)]));
                    let def = quote! {
                        #[arcorn::rewrite]
                        #derive_copy
                        pub struct #ident {
                            #(#fts),*
                        }
                    };
                    ctx.mangled_defs.insert(ident.clone(), def);
                }
                quote!(#ident)
            }
            hir::TypeKind::Tuple(ts) => {
                let ts = ts.iter().map(|t| t.lower(ctx));
                quote!((#(#ts),*))
            }
            hir::TypeKind::Vector(t) => {
                let t = t.lower(ctx);
                quote!(Vec<#t>)
            }
            hir::TypeKind::Boxed(t) => {
                let t = t.lower(ctx);
                quote!(Box<#t>)
            }
            hir::TypeKind::By(t0, t1) => {
                let t0 = t0.lower(ctx);
                let t1 = t1.lower(ctx);
                todo!()
            }
            hir::TypeKind::Unknown => unreachable!(),
            hir::TypeKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<Tokens, Context<'_>> for hir::ScalarKind {
    fn lower(&self, _ctx: &mut Context<'_>) -> Tokens {
        match self {
            Self::Bool => quote!(bool),
            Self::Char => quote!(char),
            Self::Bf16 => quote!(arcorn::bf16),
            Self::F16  => quote!(arcorn::f16),
            Self::F32  => quote!(f32),
            Self::F64  => quote!(f64),
            Self::I8   => quote!(i8),
            Self::I16  => quote!(i16),
            Self::I32  => quote!(i32),
            Self::I64  => quote!(i64),
            Self::U8   => quote!(u8),
            Self::U16  => quote!(u16),
            Self::U32  => quote!(u32),
            Self::U64  => quote!(u64),
            Self::Null => todo!(),
            Self::Str  => todo!(),
            Self::Unit => quote!(()),
            Self::Bot  => todo!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<Tokens, Context<'_>> for hir::BinOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> Tokens {
        match self.kind {
            hir::BinOpKind::Add  => quote!(+),
            hir::BinOpKind::And  => quote!(&&),
            hir::BinOpKind::Band => quote!(&),
            hir::BinOpKind::Bor  => quote!(|),
            hir::BinOpKind::Bxor => quote!(^),
            hir::BinOpKind::By   => unreachable!(),
            hir::BinOpKind::Div  => quote!(/),
            hir::BinOpKind::Equ  => quote!(==),
            hir::BinOpKind::Geq  => quote!(>=),
            hir::BinOpKind::Gt   => quote!(>),
            hir::BinOpKind::Leq  => quote!(<),
            hir::BinOpKind::Lt   => quote!(<=),
            hir::BinOpKind::Mul  => quote!(*),
            hir::BinOpKind::Mod  => quote!(%),
            hir::BinOpKind::Neq  => quote!(!=),
            hir::BinOpKind::Or   => quote!(||),
            hir::BinOpKind::Pipe => unreachable!(),
            hir::BinOpKind::Pow  => unreachable!(),
            hir::BinOpKind::Seq  => quote!(;),
            hir::BinOpKind::Sub  => quote!(-),
            hir::BinOpKind::Xor  => quote!(^),
            hir::BinOpKind::Mut  => quote!(=),
            hir::BinOpKind::Err  => unreachable!(),
        }
    }
}

impl Lower<Tokens, Context<'_>> for hir::UnOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> Tokens {
        match self.kind {
            hir::UnOpKind::Boxed => unreachable!(),
            hir::UnOpKind::Neg => quote!(-),
            hir::UnOpKind::Not => quote!(!),
            hir::UnOpKind::Err => unreachable!(),
        }
    }
}

#[rustfmt::skip]
impl Lower<Tokens, Context<'_>> for hir::LitKind {
    fn lower(&self, _ctx: &mut Context<'_>) -> Tokens {
        match self {
            Self::Bool(v) => quote!(#v),
            Self::Char(v) => quote!(#v),
            Self::Bf16(v) => {
                let v = v.to_f32();
                if v.is_nan() {
                    quote!(arcorn::bf16::NAN)
                } else if v.is_infinite() {
                    quote!(arcorn::bf16::INFINITE)
                } else {
                    quote!(arcorn::bf16::from_f32(#v))
                }
            },
            Self::F16(v) => {
                let v = v.to_f32();
                if v.is_nan() {
                    quote!(arcorn::f16::NAN)
                } else if v.is_infinite() {
                    quote!(arcorn::f16::INFINITE)
                } else {
                    quote!(arcorn::f16::from_f32(#v))
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
