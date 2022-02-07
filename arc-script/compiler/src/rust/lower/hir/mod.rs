pub(crate) mod special {
    pub(crate) mod mangle;
}

use crate::rust::Rust;
use crate::hir;
use crate::hir::HIR;
use crate::info::Info;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::lower;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Bool;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;

use proc_macro2 as pm2;
use proc_macro2::TokenStream as Code;
use quote::quote as code;

#[derive(Debug, New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    pub(crate) hir: &'i HIR,
    /// Already mangled types.
    pub(crate) mangled_names: Map<hir::Type, String>,
    pub(crate) mangled_defs: Map<syn::Ident, Code>,
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

    hir::HIR => Code {
        let defs = node.namespace.iter().map(|x| node.resolve(x).lower(ctx)).collect::<Vec<_>>();
        let mangled_defs = ctx.mangled_defs.values();
        code! {
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
                use arc_script::codegen;
                use arc_script::codegen::*;
                #(#defs)*
                #(#mangled_defs)*
            }
            pub use arc_script_output::*;
        }
    },
    hir::Item => Option<Code> {
        let item = match &node.kind {
            hir::ItemKind::TypeAlias(_)  => unreachable!(),
            hir::ItemKind::Enum(i)       => i.lower(ctx),
            hir::ItemKind::Fun(i)        => i.lower(ctx),
            hir::ItemKind::Task(i)       => i.lower(ctx),
            hir::ItemKind::ExternFun(i)  => None?,
            hir::ItemKind::ExternType(i) => None?,
            hir::ItemKind::Variant(i)    => i.lower(ctx),
        };
        Some(item)
    },
    hir::Enum => Code {
        let name = node.path.lower(ctx);
        let variants = node.variants.iter().map(|x| ctx.hir.resolve(x).lower(ctx));
        code! {
            #[codegen::rewrite]
            pub enum #name {
                #(#variants),*
            }
        }
    },
    hir::Variant => Code {
        let name = node.path.lower(ctx);
        let t = node.t.lower(ctx);
        code!(#name(#t))
    },
    hir::Block => Code {
        let stmts = node.stmts.lower(ctx);
        let v = node.var.lower(ctx);
        code!({ #(#stmts;)* #v })
    },
    VecDeque<hir::Stmt> => Vec<Code> {
        node.iter().map(|s| s.lower(ctx)).collect()
    },
    hir::Stmt => Code {
        match &node.kind {
            hir::StmtKind::Assign(l) => l.lower(ctx),
        }
    },
    hir::Assign => Code {
        let v = node.expr.lower(ctx);
        let p = node.param.lower(ctx);
        code!(let #p = #v)
    },
    hir::Fun => Code {
        let name = node.path.lower(ctx);
        let rt = node.rt.lower(ctx);
        let body = node.body.lower(ctx);
        let params = node.params.iter().map(|p| p.lower(ctx));
        match node.kind {
            hir::FunKind::Free => code! {
                pub fn #name(#(#params),*) -> #rt #body
            },
            hir::FunKind::Method => code! {
                pub fn #name(&mut self, #(#params),*) -> #rt #body
            },
        }
    },
    hir::ExternFun => Code {
        let name = node.path.lower(ctx);
        let extern_name = ctx.paths.resolve(node.path).name;
        let extern_name = extern_name.lower(ctx);
        let rt = node.rt.lower(ctx);
        let params = node.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>();
        let param_ids = node.params.iter().map(|p| p.kind.lower(ctx));
        match node.kind {
            hir::FunKind::Free => code! {
                pub fn #name(#(#params),*) -> #rt {
                    #extern_name(#(#param_ids),*)
                }
            },
            hir::FunKind::Method => code! {
                pub fn #name(&mut self, #(#params),*) -> #rt {
                    self.#extern_name(#(#param_ids),*)
                }
            },
        }
    },
    hir::Task => Code {
        let path = node.path.lower(ctx);
        let module_name = ctx.create_id(&format!("{}_mod", path));
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
            code!(#x: #t)
        });
        code! {
            #[codegen::rewrite(on_event = #on_event_fun_name, on_start = #on_start_fun_name)]
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
    hir::OnStart => (String, Code) {
        let fun = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        let name = fun.path.lower(ctx).to_string();
        let fun = fun.lower(ctx);
        (name, fun)
    },
    hir::OnEvent => (String, Code) {
        let fun = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        let name = fun.path.lower(ctx).to_string();
        let fun = fun.lower(ctx);
        (name, fun)
    },
    hir::Name => Code {
        let name = ctx.names.resolve(node);
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(name, span);
        code!(#ident)
    },
    hir::Path => Code {
        node.id.lower(ctx)
    },
    hir::PathId => Code {
        let path = ctx.resolve_to_names(node);
        let (_, path) = path.split_at(1);
        let name = path.join("_");
        let span = pm2::Span::call_site();
        let ident = syn::Ident::new(&name, span);
        code!(#ident)
    },
    Vec<hir::Param> => Vec<Code> {
        node.iter().map(|param| param.lower(ctx)).collect()
    },
    hir::Param => Code {
        let t = node.t.lower(ctx);
        let x = node.kind.lower(ctx);
        code!(#x: #t)
    },
    hir::ParamKind => Code {
        match node {
            hir::ParamKind::Ok(x)  => x.lower(ctx),
            hir::ParamKind::Ignore => code!(_),
            hir::ParamKind::Err    => unreachable!(),
        }
    },
    hir::Var => Code {
        match &node.kind {
            hir::VarKind::Ok(x, scope) => {
                let x = x.lower(ctx);
                match scope {
                    hir::ScopeKind::Local  => code!(val!(#x)),
                    hir::ScopeKind::Member => code!(((self.#x).clone())),
                    hir::ScopeKind::Global => crate::todo!("Add support for globals"),
                }
            }
            // hir::VarKind::Member => code!(node.data.#x),
            hir::VarKind::Err => unreachable!()
        }
    },
    Vec<hir::Var> => Vec<Code> {
        node.iter().map(|v| v.lower(ctx)).collect::<Vec<_>>()
    },
    hir::Expr => Code {
        match ctx.hir.exprs.resolve(node) {
            hir::ExprKind::Initialise(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                code!((self.#x).initialise(#v))
            },
            hir::ExprKind::Return(v) => {
                let v = v.lower(ctx);
                code!(return #v)
            },
            hir::ExprKind::Break(v) => {
                let v = v.lower(ctx);
                code!(break #v)
            },
            hir::ExprKind::Continue => code!(continue),
            hir::ExprKind::Access(v, f) => {
                let v = v.lower(ctx);
                let f = f.lower(ctx);
                code!(access!(#v, #f))
            }
            hir::ExprKind::Array(vs) => {
                let vs = vs.lower(ctx);
                code!([#(#vs),*])
            }
            hir::ExprKind::BinOp(v0, op, v1) => {
                let t = v0.t;
                let v0 = v0.lower(ctx);
                let v1 = v1.lower(ctx);
                if let hir::BinOpKind::Pow = &op.kind {
                    match t {
                        _ if t.is_float(ctx.info) => code!(#v0.powf(#v1)),
                        _ if t.is_int(ctx.info)   => code!(#v0.powi(#v1)),
                        x => unreachable!("{:?}", x),
                    }
                } else {
                    let op = op.lower(ctx);
                    code!(#v0 #op #v1)
                }
            },
            hir::ExprKind::Call(v, vs) => {
                let v = v.lower(ctx);
                let vs = vs.lower(ctx);
                code!(#v(#(#vs),*))
            }
            hir::ExprKind::SelfCall(x, vs) => {
                let x = x.lower(ctx);
                let vs = vs.lower(ctx);
                code!(self.#x(#(#vs),*))
            }
            hir::ExprKind::Invoke(v, x, vs) => {
                let v = v.lower(ctx);
                let x = x.lower(ctx);
                let vs = vs.lower(ctx);
                code!(#v.#x(#(#vs),*))
            }
            hir::ExprKind::Select(v, vs) => crate::todo!(),
            hir::ExprKind::Emit(v) => {
                let v = v.lower(ctx);
                code!(self.emit(#v))
            }
            hir::ExprKind::If(v, b0, b1) => {
                let v = v.lower(ctx);
                let b0 = b0.lower(ctx);
                let b1 = b1.lower(ctx);
                code!(if #v #b0 else #b1)
            }
            hir::ExprKind::Item(path) => {
                let item = &ctx.hir.resolve(path);
                match &item.kind {
                    hir::ItemKind::Task(item) => {
                        let path = item.path.lower(ctx);
                        let t = node.t.lower(ctx);
                        code!(Box::new(#path) as #t)
                    },
                    hir::ItemKind::Fun(item) => {
                        let path = path.lower(ctx);
                        match &item.kind {
                            hir::FunKind::Method => code!(self.#path),
                            hir::FunKind::Free => {
                                let t = item.t.lower(ctx);
                                code!(Box::new(#path) as #t)
                            }
                        }
                    }
                    // Do not mangle extern items
                    hir::ItemKind::ExternFun(item) => {
                        let name = ctx.paths.resolve(path).name;
                        let name = name.lower(ctx);
                        match &item.kind {
                            hir::FunKind::Method => code!(self.#name),
                            hir::FunKind::Free => {
                                let t = item.t.lower(ctx);
                                code!(Box::new(#name) as #t)
                            }
                        }
                    }
                    hir::ItemKind::ExternType(item) => {
                        let t = item.t.lower(ctx);
                        let path = path.lower(ctx);
                        code!(Box::new(#path::new) as #t)
                    }
                    _ => path.lower(ctx),
                }
            }
            hir::ExprKind::Lit(l) => l.lower(ctx),
            hir::ExprKind::Log(v) => {
                let v = v.lower(ctx);
                code!(println!("{:?}", #v))
            }
            hir::ExprKind::Loop(v) => {
                let v = v.lower(ctx);
                code!(loop { #v })
            }
            hir::ExprKind::Project(v, i) => {
                let v = v.lower(ctx);
                let i = ctx.create_ix(i.id);
                code!(#v.#i)
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
                        code!(#x: #v)
                    })
                    .collect::<Vec<_>>();
                code!(new!(#ident { #(#vfs),* }))
            }
            hir::ExprKind::Tuple(vs) => {
                let vs = vs.lower(ctx);
                code!((#(#vs),*))
            }
            hir::ExprKind::UnOp(op, v) => {
                let v = v.lower(ctx);
                let op = match &op.kind {
                    hir::UnOpKind::Neg => code!(-),
                    hir::UnOpKind::Not => code!(!),
                    hir::UnOpKind::Err => unreachable!(),
                };
                code!(#op #v)
            }
            hir::ExprKind::Enwrap(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                code!(enwrap!(#x, #v))
            }
            hir::ExprKind::Unwrap(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                code!(unwrap!(#x, #v))
            }
            hir::ExprKind::Is(x, v) => {
                let x = x.lower(ctx);
                let v = v.lower(ctx);
                code!(is!(#x, #v))
            }
            hir::ExprKind::After(v, b) => {
                let v = v.lower(ctx);
                let b = b.lower(ctx);
                code!(self.after(#v, move || #b))
            },
            hir::ExprKind::Every(v, b) => {
                let v = v.lower(ctx);
                let b = b.lower(ctx);
                code!(self.every(#v, move || #b))
            },
            hir::ExprKind::Cast(v, t) => {
                let v = v.lower(ctx);
                let t = t.lower(ctx);
                code!(#v as #t)
            },
            hir::ExprKind::Unreachable => code!(unreachable!()),
            hir::ExprKind::Err => unreachable!(),
        }
    },
    Vec<hir::Type> => Vec<Code> {
        node.iter().map(|v| v.lower(ctx)).collect::<Vec<_>>()
    },
    hir::Type => Code {
        match ctx.types.resolve(node) {
            hir::TypeKind::Array(_t, _s) => crate::todo!(),
            hir::TypeKind::Fun(ts, t) => {
                let t = t.lower(ctx);
                let ts = ts.lower(ctx);
                code!(Box<dyn ValueFn(#(#ts),*) -> #t>)
            }
            hir::TypeKind::Nominal(x) => {
                let x = x.lower(ctx);
                code!(#x)
            }
            hir::TypeKind::Scalar(s) => {
                let s = s.lower(ctx);
                code!(#s)
            }
            hir::TypeKind::Stream(t) => {
                let t = t.lower(ctx);
                code!(Stream<<#t as Convert>::T>)
            }
            hir::TypeKind::Struct(fts) => {
                let ident = node.mangle_to_ident(ctx);
                if !ctx.mangled_defs.contains_key(&ident) {
                    let fts = fts
                        .iter()
                        .map(|(f, t)| {
                            let f = f.lower(ctx);
                            let t = t.lower(ctx);
                            code!(#f : #t)
                        })
                        .collect::<Vec<_>>();
                    let def = code! {
                        #[codegen::rewrite]
                        pub struct #ident {
                            #(#fts),*
                        }
                    };
                    ctx.mangled_defs.insert(ident.clone(), def);
                }
                code!(#ident)
            }
            hir::TypeKind::Tuple(ts) => {
                let ts = ts.lower(ctx);
                code!((#(#ts),*))
            }
            hir::TypeKind::Unknown(_) => crate::ice!("Attempted to lower type variable into Code"),
            hir::TypeKind::Err => unreachable!(),
        }
    },
    hir::ScalarKind => Code {
        match node {
            hir::ScalarKind::Bool     => code!(bool),
            hir::ScalarKind::Char     => code!(char),
            hir::ScalarKind::F32      => code!(f32),
            hir::ScalarKind::F64      => code!(f64),
            hir::ScalarKind::I8       => code!(i8),
            hir::ScalarKind::I16      => code!(i16),
            hir::ScalarKind::I32      => code!(i32),
            hir::ScalarKind::I64      => code!(i64),
            hir::ScalarKind::U8       => code!(u8),
            hir::ScalarKind::U16      => code!(u16),
            hir::ScalarKind::U32      => code!(u32),
            hir::ScalarKind::U64      => code!(u64),
            hir::ScalarKind::Str      => crate::todo!(),
            hir::ScalarKind::Unit     => code!(Unit),
            hir::ScalarKind::Size     => code!(usize),
            hir::ScalarKind::DateTime => code!(u64),
            hir::ScalarKind::Duration => code!(u64),
        }
    },
    hir::BinOp => Code {
        match node.kind {
            hir::BinOpKind::Add   => code!(+),
            hir::BinOpKind::And   => code!(&&),
            hir::BinOpKind::Band  => code!(&),
            hir::BinOpKind::Bor   => code!(|),
            hir::BinOpKind::Bxor  => code!(^),
            hir::BinOpKind::Div   => code!(/),
            hir::BinOpKind::Equ   => code!(==),
            hir::BinOpKind::Geq   => code!(>=),
            hir::BinOpKind::Gt    => code!(>),
            hir::BinOpKind::In    => crate::todo!(),
            hir::BinOpKind::Leq   => code!(<),
            hir::BinOpKind::Lt    => code!(<=),
            hir::BinOpKind::Mul   => code!(*),
            hir::BinOpKind::Mod   => code!(%),
            hir::BinOpKind::Neq   => code!(!=),
            hir::BinOpKind::Or    => code!(||),
            hir::BinOpKind::Sub   => code!(-),
            hir::BinOpKind::Xor   => code!(^),
            hir::BinOpKind::Mut   => code!(=),
            hir::BinOpKind::Pow   => unreachable!(),
            hir::BinOpKind::Err   => unreachable!(),
        }
    },
    hir::LitKind => Code {
        match node {
            hir::LitKind::Bool(v)      => code!(#v),
            hir::LitKind::Char(v)      => code!(#v),
            hir::LitKind::F32(v) => match v {
                v if v.is_nan()        => code!(f32::NAN),
                v if v.is_infinite()   => code!(f32::INFINITE),
                v                      => code!(#v)
            },
            hir::LitKind::F64(v) => match v {
                v if v.is_nan()        => code!(f64::NAN),
                v if v.is_infinite()   => code!(f64::INFINITE),
                v                      => code!(#v)
            },
            hir::LitKind::I8(v)        => code!(#v),
            hir::LitKind::I16(v)       => code!(#v),
            hir::LitKind::I32(v)       => code!(#v),
            hir::LitKind::I64(v)       => code!(#v),
            hir::LitKind::U8(v)        => code!(#v),
            hir::LitKind::U16(v)       => code!(#v),
            hir::LitKind::U32(v)       => code!(#v),
            hir::LitKind::U64(v)       => code!(#v),
            hir::LitKind::Str(_v)      => crate::todo!(),
            hir::LitKind::DateTime(_v) => crate::todo!(),
            hir::LitKind::Duration(v)  => {
                let v = v.whole_seconds() as u64;
                code!(#v)
            },
            hir::LitKind::Unit         => code!(()),
            hir::LitKind::Err          => unreachable!(),
        }
    },
}
