pub(crate) mod special {
    pub(crate) mod mangle;
}

use crate::compiler::arcorn::Arcorn;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;

use arc_script_core_shared::get;
use arc_script_core_shared::lower;
use arc_script_core_shared::map;
use arc_script_core_shared::Bool;
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;
use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;
use arc_script_core_shared::VecDeque;

use proc_macro2 as pm2;
use proc_macro2::TokenStream as Rust;
use quote::quote as rust;

#[derive(Debug, New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    pub(crate) hir: &'i HIR,
    /// Already mangled types.
    pub(crate) mangled_names: Map<hir::Type, String>,
    pub(crate) mangled_defs: Map<syn::Ident, Rust>,
}

impl Context<'_> {
    /// Creates a new Rust-identifier.
    fn create_id(&self, name: &str) -> pm2::Ident {
        syn::Ident::new(name, pm2::Span::call_site())
    }
    fn create_ix(&self, index: usize) -> syn::Index {
        syn::Index::from(index)
    }
}

lower! {
    [node, ctx, hir]

    hir::HIR => Rust {
        let defs = node.namespace.iter().map(|x| node.resolve(x).lower(ctx)).collect::<Vec<_>>();
        let mangled_defs = ctx.mangled_defs.values();
        rust! {
            #[allow(non_snake_case)]
            #[allow(unused_must_use)]
            #[allow(dead_code)]
            #[allow(unused_variables)]
            #[allow(unused_imports)]
            #[allow(unused_braces)]
            #[allow(irrefutable_let_patterns)]
            #[allow(clippy::redundant_field_names)]
            #[allow(clippy::unused_unit)]
            #[allow(clippy::double_parens)]
            pub mod arc_script_output {
                use super::*;
                use arc_script::arcorn;
                #(#defs)*
                #(#mangled_defs)*
            }
            pub use arc_script_output::*;
        }
    },
    hir::Item => Option<Rust> {
        let item = match &node.kind {
            hir::ItemKind::TypeAlias(_)  => unreachable!(),
            hir::ItemKind::Enum(i)       => i.lower(ctx),
            hir::ItemKind::Fun(i)        => i.lower(ctx),
            hir::ItemKind::Task(i)       => i.lower(ctx),
            hir::ItemKind::ExternFun(i)  => None?,
            hir::ItemKind::ExternType(i) => i.lower(ctx),
            hir::ItemKind::Variant(i)    => i.lower(ctx),
        };
        Some(item)
    },
    hir::ExternType => Rust {
        let name = node.path.lower(ctx);
        rust!(use super::#name;)
    },
    hir::Enum => Rust {
        let name = node.path.lower(ctx);
        let variants = node.variants.iter().map(|x| ctx.hir.resolve(x).lower(ctx));
        rust! {
            #[arcorn::rewrite]
            pub enum #name {
                #(#variants),*
            }
        }
    },
    hir::Variant => Rust {
        let name = node.path.lower(ctx);
        let ty = node.t.lower(ctx);
        rust!(#name(#ty))
    },
    hir::Block => Rust {
        let stmts = node.stmts.lower(ctx);
        let v = node.var.lower(ctx);
        rust!({ #(#stmts;)* #v })
    },
    VecDeque<hir::Stmt> => Vec<Rust> {
        node.iter().map(|s| s.lower(ctx)).collect()
    },
    hir::Stmt => Rust {
        match &node.kind {
            hir::StmtKind::Assign(l) => l.lower(ctx),
        }
    },
    hir::Assign => Rust {
        let v = node.expr.lower(ctx);
        let p = node.param.lower(ctx);
        rust!(let #p = #v)
    },
    hir::Fun => Rust {
        let name = node.path.lower(ctx);
        let rt = node.rt.lower(ctx);
        let body = node.body.lower(ctx);
        let params = node.params.iter().map(|p| p.lower(ctx));
        match node.kind {
            hir::FunKind::Free => rust! {
                pub fn #name(#(#params),*) -> #rt #body
            },
            hir::FunKind::Method => rust! {
                pub fn #name(&mut self, #(#params),*) -> #rt #body
            },
        }
    },
    hir::ExternFun => Rust {
        let name = node.path.lower(ctx);
        let extern_name = ctx.paths.resolve(node.path).name;
        let extern_name = extern_name.lower(ctx);
        let rt = node.rt.lower(ctx);
        let params = node.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>();
        let param_ids = node.params.iter().map(|p| p.kind.lower(ctx));
        match node.kind {
            hir::FunKind::Free => rust! {
                pub fn #name(#(#params),*) -> #rt {
                    #extern_name(#(#param_ids),*)
                }
            },
            hir::FunKind::Method => rust! {
                pub fn #name(&mut self, #(#params),*) -> #rt {
                    self.#extern_name(#(#param_ids),*)
                }
            },
        }
    },
    hir::Task => Rust {
        let path = node.path.lower(ctx);
        let module_name = ctx.create_id(&format!("{}_mod", path.to_string()));
        let params = node.params.lower(ctx);
        let defs = node.namespace.iter().map(|item| ctx.hir.resolve(item).lower(ctx)).collect::<Vec<_>>();
        let ienum = ctx.hir.resolve(&node.iinterface.interior).lower(ctx);
        let oenum = ctx.hir.resolve(&node.ointerface.interior).lower(ctx);
        let (on_start_fun_name, on_start_fun) = node.on_start.lower(ctx);
        let (on_event_fun_name, on_event_fun) = node.on_event.lower(ctx);
        let cons_t = node.cons_t.lower(ctx);
        let fun_t = node.fun_t.lower(ctx);
        let fields = node.fields.iter().map(|(x, t)| {
            let x = x.lower(ctx);
            let t = t.lower(ctx);
            rust!(#x: #t)
        });
        rust! {
            #[arcorn::rewrite(on_event = #on_event_fun_name, on_start = #on_start_fun_name)]
            mod #module_name {
                struct #path {
                    #(#params,)*
                    #(#fields,)*
                }
                #ienum
                #oenum
            }
            impl #module_name::#path {
                #on_event_fun
                #on_start_fun
                #(#defs)*
            }
        }
    },
    hir::OnStart => (String, Rust) {
        let fun = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        let name = fun.path.lower(ctx).to_string();
        let fun = fun.lower(ctx);
        (name, fun)
    },
    hir::OnEvent => (String, Rust) {
        let fun = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        let name = fun.path.lower(ctx).to_string();
        let fun = fun.lower(ctx);
        (name, fun)
    },
    hir::Name => Rust {
        let name = ctx.names.resolve(node);
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(name, span);
        rust!(#ident)
    },
    hir::Path => Rust {
        node.id.lower(ctx)
    },
    hir::PathId => Rust {
        let path = ctx.resolve_to_names(node);
        let (_, path) = path.split_at(1);
        let name = path.join("_");
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(&name, span);
        rust!(#ident)
    },
    Vec<hir::Param> => Vec<Rust> {
        node.iter().map(|param| param.lower(ctx)).collect()
    },
    hir::Param => Rust {
        let t = node.t.lower(ctx);
        let x = node.kind.lower(ctx);
        rust!(#x: #t)
    },
    hir::ParamKind => Rust {
        match node {
            hir::ParamKind::Ok(x)  => x.lower(ctx),
            hir::ParamKind::Ignore => rust!(_),
            hir::ParamKind::Err    => unreachable!(),
        }
    },
    hir::Var => Rust {
        match &node.kind {
            hir::VarKind::Ok(x, scope) => {
                let x = x.lower(ctx);
                match scope {
                    hir::ScopeKind::Local  => rust!((#x.clone())),
                    hir::ScopeKind::Member => rust!(((self.#x).clone())),
                    hir::ScopeKind::Global => crate::todo!("Add support for globals"),
                }
            }
            // hir::VarKind::Member => rust!(node.data.#x),
            hir::VarKind::Err => unreachable!()
        }
    },
    Vec<hir::Var> => Vec<Rust> {
        node.iter().map(|v| v.lower(ctx)).collect::<Vec<_>>()
    },
    hir::Expr => Rust {
        match ctx.hir.exprs.resolve(node) {
            hir::ExprKind::Initialise(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                rust!((self.#x).initialise(#v))
            },
            hir::ExprKind::Return(v) => {
                let v = v.lower(ctx);
                rust!(return #v)
            },
            hir::ExprKind::Break(v) => {
                let v = v.lower(ctx);
                rust!(break #v)
            },
            hir::ExprKind::Continue => rust!(continue),
            hir::ExprKind::Access(v, f) => {
                let v = v.lower(ctx);
                let f = f.lower(ctx);
                rust!(#v.#f)
            }
            hir::ExprKind::Array(vs) => {
                let vs = vs.lower(ctx);
                rust!([#(#vs),*])
            }
            hir::ExprKind::BinOp(v0, op, v1) => {
                let t = v0.t;
                let v0 = v0.lower(ctx);
                let v1 = v1.lower(ctx);
                if let hir::BinOpKind::Pow = &op.kind {
                    match t {
                        _ if t.is_float(ctx.info) => rust!(#v0.powf(#v1)),
                        _ if t.is_int(ctx.info)   => rust!(#v0.powi(#v1)),
                        x => unreachable!("{:?}", x),
                    }
                } else {
                    let op = op.lower(ctx);
                    rust!(#v0 #op #v1)
                }
            },
            hir::ExprKind::Call(v, vs) => {
                let v = v.lower(ctx);
                let vs = vs.lower(ctx);
                rust!(#v(#(#vs),*))
            }
            hir::ExprKind::SelfCall(x, vs) => {
                let x = x.lower(ctx);
                let vs = vs.lower(ctx);
                rust!(self.#x(#(#vs),*))
            }
            hir::ExprKind::Invoke(v, x, vs) => {
                let v = v.lower(ctx);
                let x = x.lower(ctx);
                let vs = vs.lower(ctx);
                rust!(#v.#x(#(#vs),*))
            }
            hir::ExprKind::Select(v, vs) => crate::todo!(),
            hir::ExprKind::Emit(v) => {
                let v = v.lower(ctx);
                rust!(self.emit(#v))
            }
            hir::ExprKind::If(v, b0, b1) => {
                let v = v.lower(ctx);
                let b0 = b0.lower(ctx);
                let b1 = b1.lower(ctx);
                rust!(if #v #b0 else #b1)
            }
            hir::ExprKind::Item(path) => {
                let item = &ctx.hir.resolve(path);
                match &item.kind {
                    hir::ItemKind::Task(item) => {
                        let path = item.path.lower(ctx);
                        let t = node.t.lower(ctx);
                        rust!(Box::new(#path) as #t)
                    },
                    hir::ItemKind::Fun(item) => {
                        let path = path.lower(ctx);
                        match &item.kind {
                            hir::FunKind::Method => rust!(self.#path),
                            hir::FunKind::Free => {
                                let t = item.t.lower(ctx);
                                rust!(Box::new(#path) as #t)
                            }
                        }
                    }
                    // Do not mangle extern items
                    hir::ItemKind::ExternFun(item) => {
                        let name = ctx.paths.resolve(path).name;
                        let name = name.lower(ctx);
                        match &item.kind {
                            hir::FunKind::Method => rust!(self.#name),
                            hir::FunKind::Free => {
                                let t = item.t.lower(ctx);
                                rust!(Box::new(#name) as #t)
                            }
                        }
                    }
                    hir::ItemKind::ExternType(item) => {
                        let t = item.t.lower(ctx);
                        let path = path.lower(ctx);
                        rust!(Box::new(#path::new) as #t)
                    }
                    _ => path.lower(ctx),
                }
            }
            hir::ExprKind::Lit(l) => l.lower(ctx),
            hir::ExprKind::Log(v) => {
                let v = v.lower(ctx);
                rust!(println!("{:?}", #v))
            }
            hir::ExprKind::Loop(v) => {
                let v = v.lower(ctx);
                rust!(loop { #v })
            }
            hir::ExprKind::Project(v, i) => {
                let v = v.lower(ctx);
                let i = ctx.create_ix(i.id);
                rust!(#v.#i)
            }
            hir::ExprKind::Struct(vfs) => {
                // NOTE: Lower here to ensure that the type definition is interned.
                node.t.lower(ctx);
                let ident = node.t.mangle_to_ident(ctx);
                let vfs = vfs
                    .iter()
                    .map(|(x, v)| {
                        let x = x.lower(ctx);
                        let v = v.lower(ctx);
                        rust!(#x: #v)
                    })
                    .collect::<Vec<_>>();
                rust!(#ident { #(#vfs),* })
            }
            hir::ExprKind::Tuple(vs) => {
                let vs = vs.lower(ctx);
                rust!((#(#vs),*))
            }
            hir::ExprKind::UnOp(op, v) => {
                let v = v.lower(ctx);
                let op = match &op.kind {
                    hir::UnOpKind::Neg => rust!(-),
                    hir::UnOpKind::Not => rust!(!),
                    hir::UnOpKind::Err => unreachable!(),
                };
                rust!(#op #v)
            }
            hir::ExprKind::Enwrap(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                rust!(arcorn::enwrap!(#x, #v))
            }
            hir::ExprKind::Unwrap(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                rust!(arcorn::unwrap!(#x, #v))
            }
            hir::ExprKind::Is(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                rust!(arcorn::is!(#x, #v))
            }
            hir::ExprKind::After(v, b) => {
                let v = v.lower(ctx);
                let b = b.lower(ctx);
                rust!(self.after(#v, move || #b))
            },
            hir::ExprKind::Every(v, b) => {
                let v = v.lower(ctx);
                let b = b.lower(ctx);
                rust!(self.every(#v, move || #b))
            },
            hir::ExprKind::Cast(v, t) => {
                let v = v.lower(ctx);
                let t = t.lower(ctx);
                rust!(#v as #t)
            },
            hir::ExprKind::Unreachable => rust!(unreachable!()),
            hir::ExprKind::Err => unreachable!(),
        }
    },
    Vec<hir::Type> => Vec<Rust> {
        node.iter().map(|v| v.lower(ctx)).collect::<Vec<_>>()
    },
    hir::Type => Rust {
        match ctx.types.resolve(node) {
            hir::TypeKind::Array(_t, _s) => crate::todo!(),
            hir::TypeKind::Fun(ts, t) => {
                let t = t.lower(ctx);
                let ts = ts.lower(ctx);
                rust!(Box<dyn arcorn::ArcornFn(#(#ts),*) -> #t>)
            }
            hir::TypeKind::Nominal(x) => {
                let x = x.lower(ctx);
                rust!(#x)
            }
            hir::TypeKind::Scalar(s) => {
                let s = s.lower(ctx);
                rust!(#s)
            }
            hir::TypeKind::Stream(t) => {
                let t = t.lower(ctx);
                rust!(arcorn::Stream<#t>)
            }
            hir::TypeKind::Struct(fts) => {
                let ident = node.mangle_to_ident(ctx);
                if !ctx.mangled_defs.contains_key(&ident) {
                    let fts = fts
                        .iter()
                        .map(|(f, t)| {
                            let f = f.lower(ctx);
                            let t = t.lower(ctx);
                            rust!(#f : #t)
                        })
                        .collect::<Vec<_>>();
                    let def = rust! {
                        #[arcorn::rewrite]
                        pub struct #ident {
                            #(#fts),*
                        }
                    };
                    ctx.mangled_defs.insert(ident.clone(), def);
                }
                rust!(#ident)
            }
            hir::TypeKind::Tuple(ts) => {
                let ts = ts.lower(ctx);
                rust!((#(#ts),*))
            }
            hir::TypeKind::Unknown(_) => crate::ice!("Attempted to lower type variable into Rust"),
            hir::TypeKind::Err => unreachable!(),
        }
    },
    hir::ScalarKind => Rust {
        match node {
            hir::ScalarKind::Bool     => rust!(bool),
            hir::ScalarKind::Char     => rust!(char),
            hir::ScalarKind::F32      => rust!(f32),
            hir::ScalarKind::F64      => rust!(f64),
            hir::ScalarKind::I8       => rust!(i8),
            hir::ScalarKind::I16      => rust!(i16),
            hir::ScalarKind::I32      => rust!(i32),
            hir::ScalarKind::I64      => rust!(i64),
            hir::ScalarKind::U8       => rust!(u8),
            hir::ScalarKind::U16      => rust!(u16),
            hir::ScalarKind::U32      => rust!(u32),
            hir::ScalarKind::U64      => rust!(u64),
            hir::ScalarKind::Str      => crate::todo!(),
            hir::ScalarKind::Unit     => rust!(()),
            hir::ScalarKind::Size     => rust!(usize),
            hir::ScalarKind::DateTime => rust!(u64),
            hir::ScalarKind::Duration => rust!(u64),
        }
    },
    hir::BinOp => Rust {
        match node.kind {
            hir::BinOpKind::Add   => rust!(+),
            hir::BinOpKind::And   => rust!(&&),
            hir::BinOpKind::Band  => rust!(&),
            hir::BinOpKind::Bor   => rust!(|),
            hir::BinOpKind::Bxor  => rust!(^),
            hir::BinOpKind::Div   => rust!(/),
            hir::BinOpKind::Equ   => rust!(==),
            hir::BinOpKind::Geq   => rust!(>=),
            hir::BinOpKind::Gt    => rust!(>),
            hir::BinOpKind::In    => crate::todo!(),
            hir::BinOpKind::Leq   => rust!(<),
            hir::BinOpKind::Lt    => rust!(<=),
            hir::BinOpKind::Mul   => rust!(*),
            hir::BinOpKind::Mod   => rust!(%),
            hir::BinOpKind::Neq   => rust!(!=),
            hir::BinOpKind::Or    => rust!(||),
            hir::BinOpKind::Sub   => rust!(-),
            hir::BinOpKind::Xor   => rust!(^),
            hir::BinOpKind::Mut   => rust!(=),
            hir::BinOpKind::Pow   => unreachable!(),
            hir::BinOpKind::Err   => unreachable!(),
        }
    },
    hir::LitKind => Rust {
        match node {
            hir::LitKind::Bool(v)      => rust!(#v),
            hir::LitKind::Char(v)      => rust!(#v),
            hir::LitKind::F32(v) => match v {
                v if v.is_nan()        => rust!(f32::NAN),
                v if v.is_infinite()   => rust!(f32::INFINITE),
                v                      => rust!(#v)
            },
            hir::LitKind::F64(v) => match v {
                v if v.is_nan()        => rust!(f64::NAN),
                v if v.is_infinite()   => rust!(f64::INFINITE),
                v                      => rust!(#v)
            },
            hir::LitKind::I8(v)        => rust!(#v),
            hir::LitKind::I16(v)       => rust!(#v),
            hir::LitKind::I32(v)       => rust!(#v),
            hir::LitKind::I64(v)       => rust!(#v),
            hir::LitKind::U8(v)        => rust!(#v),
            hir::LitKind::U16(v)       => rust!(#v),
            hir::LitKind::U32(v)       => rust!(#v),
            hir::LitKind::U64(v)       => rust!(#v),
            hir::LitKind::Str(_v)      => crate::todo!(),
            hir::LitKind::DateTime(_v) => crate::todo!(),
            hir::LitKind::Duration(v)  => {
                let v = v.whole_seconds() as u64;
                rust!(#v)
            },
            hir::LitKind::Unit         => rust!(()),
            hir::LitKind::Err          => unreachable!(),
        }
    },
}
