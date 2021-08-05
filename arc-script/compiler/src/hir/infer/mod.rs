use crate::hir;
use crate::hir::infer::specialise::Specialise;
use crate::hir::infer::unify::Unify;
use crate::hir::utils::SortFields;
use crate::info::diags::Diagnostic;
use crate::info::diags::Error;
use crate::info::diags::Warning;
use crate::info::files::Loc;
use crate::info::types::Type;
use crate::info::Info;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::VecMap;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::OrdMap;
use arc_script_compiler_shared::Shrinkwrap;

pub(crate) mod equality;
pub(crate) mod subtype;
pub(crate) mod unify;
pub(crate) mod hash;
pub(crate) mod specialise;
// pub(crate) mod generalise;

use tracing::instrument;

/// Context used during type inference.
#[derive(New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    /// Polymorphic types (Input to type inference).
    hir: &'i hir::HIR,
    mono: &'i mut hir::HIR,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    /// Environment which stores the types of variables. As all variables have unique names there
    /// is no need to keep scopes around.
    env: &'i mut Map<hir::Name, hir::Type>,
    /// Instantiated items and their types.
    instances: &'i mut Map<hir::Path, hir::Type>,
    /// Path to output interface enum, which a task is expected to emit.
    ointerface_interior: Option<hir::Path>,
    iinterface_interior: Option<hir::Path>,
    loc: Loc,
}

impl hir::HIR {
    /// Infers the types of all type variables in the HIR.
    #[instrument(name = "(Infer)", level = "debug", skip(self, info))]
    pub(crate) fn infer(&self, info: &mut Info) {
        let env = &mut Map::default();
        let mut mono = hir::HIR::default();
        let instances = &mut Map::default();
        let ctx = &mut Context::new(self, &mut mono, info, env, instances, None, None, Loc::Fake);
        for item in &self.namespace {
            self.resolve(item).infer(ctx);
        }
    }
}

pub(crate) trait Infer<'i> {
    fn infer(&self, ctx: &mut Context<'i>);
}

use crate::info::modes::Verbosity;
/// Macro for implementing the `Infer` trait.
macro_rules! infer {
    {
        [$node:ident, $ctx:ident]
        $($ty:path => $expr:expr ,)*
    } => {
        $(
            impl Infer<'_> for $ty {
                fn infer(&self, $ctx: &mut Context<'_>) {
                    let $node = self;
                    tracing::trace!("⊢ {:<14}: {}", stringify!($ty), $ctx.hir.pretty($node, &$ctx.info));
                    $expr;
                }
            }
        )*
    };
}

impl Context<'_> {
    fn params_to_fields(&mut self, params: &[hir::Param]) -> VecMap<hir::Name, hir::Type> {
        params
            .iter()
            .filter_map(|p| match &p.kind {
                hir::ParamKind::Ok(x) => Some((*x, p.t)),
                hir::ParamKind::Ignore => None,
                hir::ParamKind::Err => None,
            })
            .collect()
    }
    fn assignments_to_fields(
        &mut self,
        assignments: &[hir::Assign],
    ) -> VecMap<hir::Name, hir::Type> {
        assignments
            .iter()
            .filter_map(|l| match &l.param.kind {
                hir::ParamKind::Ok(x) => Some((*x, l.param.t)),
                hir::ParamKind::Ignore => None,
                hir::ParamKind::Err => None,
            })
            .collect()
    }
}

infer! {
    [node, ctx]

    hir::Item => {
        // When encountering an item, we must *generalise* it by
        // replacing all its free type variables with quantifiers
        match &node.kind {
            hir::ItemKind::Fun(item)        => item.infer(ctx),
            hir::ItemKind::Task(item)       => item.infer(ctx),
            hir::ItemKind::ExternFun(item)  => item.infer(ctx),
            hir::ItemKind::TypeAlias(_)     => {}
            hir::ItemKind::Enum(_)          => {}
            hir::ItemKind::ExternType(item) => item.infer(ctx),
            hir::ItemKind::Variant(_)       => {}
        }
    },
    // Items are assumed to be generalised before type inference starts.
    //
    // When encountering an item, we must
    // 1. Clone it
    // 2. Replace all types with fresh type variables (*specialise*)
    // 3. Replace the polymorphic item with the monomorphic item temporarily
    //   3. Or maybe, add it to the stack of inferred items
    // 4. Infer the type of the item
    // 5. Store it in a map of (eventually) monomorphised items
    hir::Fun => {
        // let fun = node.specialise(ctx);
        let ts = node.params.iter().map(|x| x.t).collect();
        let t = node.body.var.t;
        ctx.unify(node.t, hir::TypeKind::Fun(ts, t));
        ctx.unify(node.rt, t);
        for param in &node.params {
            if let hir::ParamKind::Ok(x) = &param.kind {
                ctx.env.insert(*x, param.t);
            }
        }
        node.body.infer(ctx);
    },
    hir::ExternFun => {
        // let fun = node.specialise(ctx);
        let ts = node.params.iter().map(|x| x.t).collect();
        let t = ctx.types.fresh();
        ctx.unify(node.t, hir::TypeKind::Fun(ts, t));
        ctx.unify(node.rt, t);
    },
    hir::ExternType => {
        // let fun = node.specialise(ctx);
        let ts = node.params.iter().map(|x| x.t).collect();
        let t = ctx.types.intern(hir::TypeKind::Nominal(node.path));
        ctx.unify(node.t, hir::TypeKind::Fun(ts, t));
    },
    hir::Block => {
        for stmt in &node.stmts {
            stmt.infer(ctx);
        }
        node.var.infer(ctx);
    },
    hir::Task => {
        // Input key-type of this task
        let ikey_t = ctx.types.fresh();
        // Infer parameters
        for param in &node.params {
            if let hir::ParamKind::Ok(x) = &param.kind {
                ctx.env.insert(*x, param.t);
            }
        }
        // Infer input/output streams/events
        node.iinterface.infer(ctx);
        node.ointerface.infer(ctx);
        // All keys of the input-streams must match
        node.iinterface.keys.iter().for_each(|t| ctx.unify(*t, ikey_t));
        // Infer state-fields
        node.fields.iter().for_each(|(x, t)| {
            ctx.env.insert(*x, *t);
        });
        // Infer handler functions
        ctx.iinterface_interior = Some(node.iinterface.interior);
        ctx.ointerface_interior = Some(node.ointerface.interior);
        node.on_start.infer(ctx);
        node.on_event.infer(ctx);
        ctx.iinterface_interior = None;
        ctx.ointerface_interior = None;
        // Infer task items
        for item in &node.namespace {
            ctx.hir.resolve(item).infer(ctx);
        }
        // Construct a function-type for calling the task.
        let ts = node.params.iter().map(|x| x.t).collect();
        let it = node.iinterface.exterior.clone();
        let ot = match node.ointerface.exterior.as_slice() {
            [] => ctx.types.intern(hir::ScalarKind::Unit),
            [t] => *t,
            ts => ctx.types.intern(hir::TypeKind::Tuple(ts.to_vec())),
        };
        // Construct type for connecting the task to a stream.
        ctx.unify(node.fun_t, hir::TypeKind::Fun(it, ot));
        ctx.unify(node.cons_t, hir::TypeKind::Fun(ts, node.fun_t));
        let mut fs = ctx.params_to_fields(&node.params);
        fs.extend(node.fields.clone());
        ctx.unify(node.struct_t, hir::TypeKind::Struct(fs));
    },
    hir::Interface => {
        let val_x = ctx.names.common.val;
        let key_x = ctx.names.common.key;
        let item = get!(&ctx.hir.resolve(&node.interior).kind, hir::ItemKind::Enum(x));
        for (exterior_t, interior_x) in node.exterior.clone().into_iter().zip(&item.variants) {
            let interior_t = get!(&ctx.hir.resolve(interior_x).kind, hir::ItemKind::Variant(x)).t;
            let key_t = ctx.types.fresh();
            let val_t = ctx.types.fresh();
            ctx.unify(interior_t, vec![(key_x, key_t), (val_x, val_t)]);
            ctx.unify(exterior_t, hir::TypeKind::Stream(interior_t));
        }
    },
    hir::OnStart => {
        let item = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        item.infer(ctx);
        ctx.unify(item.body.var.t, hir::ScalarKind::Unit);
    },
    hir::OnEvent => {
        let item = get!(&ctx.hir.resolve(node.fun).kind, hir::ItemKind::Fun(x));
        item.infer(ctx);
        ctx.unify(item.body.var.t, hir::ScalarKind::Unit);
        ctx.unify(item.params[0].t, hir::TypeKind::Nominal(ctx.iinterface_interior.unwrap()));
    },
    hir::Assign => {
        node.expr.infer(ctx);
        match &node.param.kind {
            hir::ParamKind::Ok(x) => {
                ctx.env.insert(*x, node.param.t);
                ctx.unify(node.expr.t, node.param.t);
            }
            hir::ParamKind::Ignore => {}
            hir::ParamKind::Err => {}
        }
    },
    hir::Var => if let hir::VarKind::Ok(x, _) = &node.kind {
        if let Some(t) = ctx.env.get(x).copied() {
            ctx.unify(node.t, t);
        }
    },
    hir::Expr => {
        use hir::BinOpKind::*;
        use hir::ScalarKind::*;
        let loc = ctx.loc;
        ctx.loc = node.loc;
        match ctx.hir.exprs.resolve(node) {
            hir::ExprKind::Item(x) => {
                match &ctx.hir.resolve(x).kind {
                    hir::ItemKind::Fun(item)        => ctx.unify(node.t, item.t),
                    hir::ItemKind::Task(item)       => ctx.unify(node.t, item.cons_t),
                    hir::ItemKind::ExternFun(item)  => ctx.unify(node.t, item.t),
                    hir::ItemKind::ExternType(item) => ctx.unify(node.t, item.t),
                    hir::ItemKind::TypeAlias(_)     => unreachable!(),
                    hir::ItemKind::Enum(_)          => unreachable!(),
                    hir::ItemKind::Variant(_)       => unreachable!(),
                }
            },
            hir::ExprKind::Lit(kind) => {
                let kind = match kind {
                    hir::LitKind::I8(_)       => I8,
                    hir::LitKind::I16(_)      => I16,
                    hir::LitKind::I32(_)      => I32,
                    hir::LitKind::I64(_)      => I64,
                    hir::LitKind::U8(_)       => U8,
                    hir::LitKind::U16(_)      => U16,
                    hir::LitKind::U32(_)      => U32,
                    hir::LitKind::U64(_)      => U64,
                    hir::LitKind::F32(_)      => F32,
                    hir::LitKind::F64(_)      => F64,
                    hir::LitKind::Bool(_)     => Bool,
                    hir::LitKind::Unit        => Unit,
                    hir::LitKind::DateTime(_) => DateTime,
                    hir::LitKind::Duration(_) => Duration,
                    hir::LitKind::Char(_)     => Char,
                    hir::LitKind::Str(_)      => Str,
                    hir::LitKind::Err         => return,
                };
                ctx.unify(node.t, kind);
            }
            hir::ExprKind::Array(vs) => {
                let elem_t = ctx.types.fresh();
                let dim = hir::Dim::new(hir::DimKind::Val(vs.len() as i32));
                vs.iter().for_each(|v| ctx.unify(elem_t, v.t));
                let shape = hir::Shape::new(vec![dim]);
                ctx.unify(node.t, hir::TypeKind::Array(elem_t, shape));
                vs.infer(ctx);
            }
            // NOTE: We sort fields-types by field name.
            hir::ExprKind::Struct(fs) => {
                fs.infer(ctx);
                let fs = fs.iter().map(|(x, v)| (*x, v.t)).collect::<VecMap<_, _>>().sort_fields(ctx.info);
                ctx.unify(node.t, hir::TypeKind::Struct(fs));
            }
            hir::ExprKind::Enwrap(x0, v) => {
                v.infer(ctx);
                let item = &ctx.hir.resolve(x0).kind;
                let item = get!(item, hir::ItemKind::Variant(x));
                let x1 = ctx.paths.resolve(x0).pred.unwrap().into();
                ctx.unify(v.t, item.t);
                ctx.unify(node.t, hir::TypeKind::Nominal(x1));
            }
            hir::ExprKind::Unwrap(x0, v) => {
                v.infer(ctx);
                let variant = get!(&ctx.hir.resolve(x0).kind, hir::ItemKind::Variant(x));
                let x1 = ctx.paths.resolve(x0).pred.unwrap().into();
                ctx.unify(v.t, hir::TypeKind::Nominal(x1));
                ctx.unify(node.t, variant.t);
            }
            hir::ExprKind::Is(x0, v) => {
                v.infer(ctx);
                let x1 = ctx.paths.resolve(x0).pred.unwrap().into();
                ctx.unify(v.t, hir::TypeKind::Nominal(x1));
                ctx.unify(node.t, Bool);
            }
            hir::ExprKind::Tuple(vs) => {
                let ts = vs.iter().map(|v| v.t).collect();
                ctx.unify(node.t, hir::TypeKind::Tuple(ts));
                vs.infer(ctx);
            }
            hir::ExprKind::BinOp(v0, op, v1) => {
                v0.infer(ctx);
                v1.infer(ctx);
                match &op.kind {
                    Add | Div | Mul | Sub | Mod | Pow => {
                        ctx.unify(node.t, v0.t);
                        ctx.unify(node.t, v1.t);
                    }
                    Equ | Neq | Gt | Lt | Geq | Leq => {
                        ctx.unify(v0.t, v1.t);
                        ctx.unify(node.t, Bool);
                    }
                    Or | And | Xor => {
                        ctx.unify(node.t, v0.t);
                        ctx.unify(node.t, v1.t);
                        ctx.unify(node.t, Bool);
                    }
                    Band | Bor | Bxor => {
                        ctx.unify(node.t, v0.t);
                        ctx.unify(node.t, v1.t);
                    }
                    Mut => {
                        ctx.unify(node.t, Unit);
                        ctx.unify(v0.t, v1.t);
                    }
                    In => ctx.unify(node.t, Bool),
                    hir::BinOpKind::Err => {}
                }
            }
            hir::ExprKind::UnOp(op, v) => {
                v.infer(ctx);
                match &op.kind {
                    hir::UnOpKind::Not => {
                        ctx.unify(node.t, v.t);
                        ctx.unify(v.t, Bool);
                    }
                    hir::UnOpKind::Neg => ctx.unify(node.t, v.t),
                    hir::UnOpKind::Err => {},
                }
            },
            hir::ExprKind::Call(v, vs) => {
                v.infer(ctx);
                vs.infer(ctx);
                let ts = vs.iter().map(|v| v.t).collect();
                ctx.unify(v.t, hir::TypeKind::Fun(ts, node.t));
            },
            hir::ExprKind::SelfCall(x, vs) => {
                let item = get!(&ctx.hir.resolve(x).kind, hir::ItemKind::Fun(item));
                vs.infer(ctx);
                let ts = vs.iter().map(|v| v.t).collect();
                ctx.unify(item.t, hir::TypeKind::Fun(ts, node.t));
            },
            hir::ExprKind::Invoke(v, x, vs) => {
                v.infer(ctx);
                vs.infer(ctx);
                // TODO: Currently we require the type of the object to be known
                match &ctx.types.resolve(v.t) {
                    hir::TypeKind::Nominal(object_x) => {
                        let method_x = ctx.paths.intern_child(object_x, *x);
                        if let hir::ItemKind::ExternFun(item) = &ctx.hir.resolve(method_x).kind {
                            let ts = vs.iter().map(|v| v.t).collect();
                            ctx.unify(item.t, hir::TypeKind::Fun(ts, node.t));
                        } else {
                            crate::todo!("Expected function");
                        }
                    }
                    _ => {
                        ctx.diags.intern(Error::TypeMustBeKnownAtThisPoint { loc: node.loc });
                    },
                }
            },
            hir::ExprKind::Select(v, vs) => crate::todo!(),
            hir::ExprKind::Project(v, i) => {
                v.infer(ctx);
                if let hir::TypeKind::Tuple(ts) = ctx.types.resolve(v.t) {
                    if let Some(t) = ts.get(i.id) {
                        ctx.unify(node.t, *t);
                    } else {
                        ctx.diags.intern(Error::OutOfBoundsProject { loc: node.loc })
                    }
                }
            }
            hir::ExprKind::Access(v, x) => {
                v.infer(ctx);
                if let hir::TypeKind::Struct(fs) = ctx.types.resolve(v.t) {
                    if let Some(t) = fs.get(x) {
                        ctx.unify(node.t, *t);
                    } else {
                        ctx.diags.intern(Error::FieldNotFound { loc: node.loc })
                    }
                }
            }
            hir::ExprKind::Emit(v) => {
                ctx.unify(node.t, Unit);
                ctx.unify(v.t, hir::TypeKind::Nominal(ctx.ointerface_interior.unwrap()));
                v.infer(ctx);
            }
            hir::ExprKind::Log(v) => {
                ctx.unify(node.t, Unit);
                v.infer(ctx);
            }
            hir::ExprKind::If(v, b0, b1) => {
                ctx.unify(v.t, Bool);
                ctx.unify(node.t, b0.var.t);
                ctx.unify(node.t, b1.var.t);
                v.infer(ctx);
                b0.infer(ctx);
                b1.infer(ctx);
            }
            hir::ExprKind::Loop(_) => crate::todo!(),
            hir::ExprKind::After(v, b) | hir::ExprKind::Every(v, b) => {
                ctx.unify(v.t, Duration);
                ctx.unify(node.t, Unit);
                ctx.unify(node.t, b.var.t);
                b.infer(ctx);
                v.infer(ctx);
            },
            hir::ExprKind::Cast(_, _) => {},
            hir::ExprKind::Unreachable => {},
            hir::ExprKind::Initialise(x, v) => {
                let t = ctx.env.get(x).copied().unwrap();
                ctx.unify(v.t, t);
                ctx.unify(node.t, Unit);
                v.infer(ctx);
            },
            hir::ExprKind::Return(v) => {
                ctx.unify(node.t, Unit);
                v.infer(ctx);
            },
            hir::ExprKind::Break(v) => {
                ctx.unify(node.t, Unit);
                v.infer(ctx);
            },
            hir::ExprKind::Continue => {
                ctx.unify(node.t, Unit);
            },
            hir::ExprKind::Err => {}
        }
        ctx.loc = loc;
    },
    hir::Stmt => {
        match &node.kind {
            hir::StmtKind::Assign(item) => item.infer(ctx),
        }
    },
    Vec<hir::Var> => {
        node.iter().for_each(|v| v.infer(ctx))
    },
    VecMap<hir::Name, hir::Var> => {
        node.values().for_each(|v| v.infer(ctx))
    },
}
