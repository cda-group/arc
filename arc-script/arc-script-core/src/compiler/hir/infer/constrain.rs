use crate::compiler::hir::infer::unify::Unify;
use crate::compiler::hir::{
    self, BinOpKind, Dim, DimKind, Expr, ExprKind, Extern, Fun, Item, ItemKind, LitKind, Param,
    ParamKind, Path, ScalarKind, Shape, State, Task, Type, TypeKind, UnOpKind, HIR,
};
use crate::compiler::info::diags::{Diagnostic, Error, Warning};
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::map;
use arc_script_core_shared::VecMap;

use super::Context;

use arc_script_core_shared::get;

pub(crate) trait Constrain<'i> {
    fn constrain(&self, ctx: &mut Context<'i>);
}

impl Constrain<'_> for Item {
    #[rustfmt::skip]
    fn constrain(&self, ctx: &mut Context<'_>) {
        match &self.kind {
            ItemKind::Fun(item)   => item.constrain(ctx),
            ItemKind::Task(item)  => item.constrain(ctx),
            ItemKind::Alias(_)    => {}
            ItemKind::Enum(_)     => {}
            ItemKind::Extern(_)   => {}
            ItemKind::Variant(_)  => {}
        }
    }
}

impl Constrain<'_> for Fun {
    fn constrain(&self, ctx: &mut Context<'_>) {
        let tvs = self.params.iter().map(|x| x.tv).collect();
        let tv = self.body.tv;
        ctx.unify(self.tv, TypeKind::Fun(tvs, tv));
        ctx.unify(self.rtv, tv);
        for p in &self.params {
            if let ParamKind::Var(x) = &p.kind {
                ctx.env.insert(*x, *p);
            }
        }
        self.body.constrain(ctx);
    }
}

impl Constrain<'_> for Task {
    /// On the outside, a task is nothing more than a function which returns a function.
    fn constrain(&self, ctx: &mut Context<'_>) {
        self.params.iter().for_each(|p| {
            if let ParamKind::Var(x) = &p.kind {
                ctx.env.insert(*x, *p);
            }
        });
        self.states.iter().for_each(|s| {
            let p = &s.param;
            ctx.unify(p.tv, s.init.tv);
            if let ParamKind::Var(x) = &p.kind {
                ctx.env.insert(*x, *p);
            }
        });
        if let Some(on) = &self.on {
            if let ParamKind::Var(x) = &on.param.kind {
                ctx.env.insert(*x, on.param);
            }
            on.body.constrain(ctx);
            ctx.unify(on.param.tv, self.ihub.tv);
        }
        self.items.iter().for_each(|item| {
            let item = ctx.defs.get(item).unwrap();
            item.constrain(ctx)
        });
        let tvs = self.params.iter().map(|x| x.tv).collect();
        let itvs = self.ihub.constrain(ctx);
        let otvs = self.ohub.constrain(ctx);
        let otv = match otvs.len() {
            0 => ctx.info.types.intern(ScalarKind::Unit),
            1 => otvs[0],
            _ => ctx.info.types.intern(TypeKind::Tuple(otvs)),
        };
        let ttv = ctx.info.types.intern(TypeKind::Fun(itvs, otv));
        ctx.unify(self.tv, TypeKind::Fun(tvs, ttv));
    }
}

impl hir::Hub {
    /// Constrains the internal port types of a hub and returns the external port types
    fn constrain(&self, ctx: &mut Context<'_>) -> Vec<TypeId> {
        match &self.kind {
            hir::HubKind::Tagged(x) => {
                ctx.unify(self.tv, TypeKind::Nominal(*x));
                let item = ctx.defs.get(x).unwrap();
                let item = get!(&item.kind, hir::ItemKind::Enum(x));
                item.variants
                    .iter()
                    .map(|x| {
                        let item = ctx.defs.get(x).unwrap();
                        get!(&item.kind, hir::ItemKind::Variant(x)).tv
                    })
                    .collect::<Vec<_>>()
            }
            hir::HubKind::Single(tv) => {
                ctx.unify(*tv, hir::TypeKind::Stream(self.tv));
                vec![*tv]
            }
        }
    }
}

impl Constrain<'_> for State {
    fn constrain(&self, ctx: &mut Context<'_>) {
        self.init.constrain(ctx);
        ctx.unify(self.param.tv, self.init.tv);
    }
}

impl Constrain<'_> for Expr {
    /// Constrains an expression based on its subexpressions.
    fn constrain(&self, ctx: &mut Context<'_>) {
        use BinOpKind::*;
        use ScalarKind::*;
        let loc = ctx.loc;
        ctx.loc = self.loc;
        match &self.kind {
            ExprKind::Let(p, e0, e1) => {
                if let ParamKind::Var(x) = p.kind {
                    ctx.env.insert(x, *p);
                    ctx.unify(p.tv, e0.tv);
                    ctx.unify(self.tv, e1.tv);
                    e0.constrain(ctx);
                    e1.constrain(ctx);
                }
            }
            ExprKind::Var(x, _) => {
                if let Some(x) = ctx.env.get(x).cloned() {
                    ctx.unify(self.tv, x.tv);
                }
            }
            #[rustfmt::skip]
            ExprKind::Item(x) => {
                match &ctx.defs.get(x).unwrap().kind {
                    ItemKind::Fun(item)    => ctx.unify(self.tv, item.tv),
                    ItemKind::Task(item)   => ctx.unify(self.tv, item.tv),
                    ItemKind::Extern(item) => ctx.unify(self.tv, item.tv),
                    ItemKind::Alias(_)     => unreachable!(),
                    ItemKind::Enum(_)      => unreachable!(),
                    ItemKind::Variant(_)   => unreachable!(),
                }
            },
            #[rustfmt::skip]
            ExprKind::Lit(kind) => {
                let kind = match kind {
                    LitKind::I8(_)   => I8,
                    LitKind::I16(_)  => I16,
                    LitKind::I32(_)  => I32,
                    LitKind::I64(_)  => I64,
                    LitKind::U8(_)   => U8,
                    LitKind::U16(_)  => U16,
                    LitKind::U32(_)  => U32,
                    LitKind::U64(_)  => U64,
                    LitKind::Bf16(_) => Bf16,
                    LitKind::F16(_)  => F16,
                    LitKind::F32(_)  => F32,
                    LitKind::F64(_)  => F64,
                    LitKind::Bool(_) => Bool,
                    LitKind::Unit    => Unit,
                    LitKind::Time(_) => todo!(),
                    LitKind::Char(_) => Char,
                    LitKind::Str(_)  => Str,
                    LitKind::Err     => return,
                };
                ctx.unify(self.tv, kind);
            }
            ExprKind::Array(es) => {
                let elem_tv = ctx.info.types.fresh();
                let dim = Dim::new(DimKind::Val(es.len() as i32));
                es.iter().for_each(|e| ctx.unify(elem_tv, e.tv));
                let shape = Shape::new(vec![dim]);
                ctx.unify(self.tv, TypeKind::Array(elem_tv, shape));
                es.constrain(ctx);
            }
            ExprKind::Struct(fs) => {
                let fields = fs
                    .iter()
                    .map(|(field, arg)| (*field, arg.tv))
                    .collect::<VecMap<_, _>>();
                ctx.unify(self.tv, TypeKind::Struct(fields));
                fs.constrain(ctx);
            }
            ExprKind::Enwrap(x0, e) => {
                e.constrain(ctx);
                let item = &ctx.defs.get(x0).unwrap().kind;
                let item = get!(item, ItemKind::Variant(x));
                let x1 = ctx.info.paths.resolve(x0.id).pred.unwrap().into();
                ctx.unify(e.tv, item.tv);
                ctx.unify(self.tv, TypeKind::Nominal(x1));
            }
            ExprKind::Unwrap(x0, e) => {
                e.constrain(ctx);
                let item = ctx.defs.get(x0).unwrap();
                let item = get!(&item.kind, ItemKind::Variant(x));
                let x1 = ctx.info.paths.resolve(x0.id).pred.unwrap().into();
                ctx.unify(e.tv, TypeKind::Nominal(x1));
                ctx.unify(self.tv, item.tv);
            }
            ExprKind::Is(x0, e) => {
                e.constrain(ctx);
                let item = ctx.defs.get(x0).unwrap();
                let _item = get!(&item.kind, ItemKind::Variant(x));
                let x1 = ctx.info.paths.resolve(x0.id).pred.unwrap().into();
                ctx.unify(e.tv, TypeKind::Nominal(x1));
                ctx.unify(self.tv, Bool);
            }
            ExprKind::Tuple(es) => {
                let tvs = es.iter().map(|arg| arg.tv).collect();
                ctx.unify(self.tv, TypeKind::Tuple(tvs));
                es.constrain(ctx);
            }
            ExprKind::BinOp(e0, op, e1) => {
                e0.constrain(ctx);
                e1.constrain(ctx);
                match &op.kind {
                    Add | Div | Mul | Sub | Mod => {
                        ctx.unify(e0.tv, e1.tv);
                        ctx.unify(self.tv, e1.tv);
                    }
                    Pow => {
                        ctx.unify(self.tv, e0.tv);
                        if let TypeKind::Scalar(kind) = ctx.info.types.resolve(e0.tv).kind {
                            match kind {
                                I8 | I16 | I32 | I64 => ctx.unify(e1.tv, I32),
                                F32 => ctx.unify(e1.tv, F32),
                                F64 => ctx.unify(e1.tv, F64),
                                _ => {}
                            }
                        }
                        if let TypeKind::Scalar(kind) = ctx.info.types.resolve(e1.tv).kind {
                            match kind {
                                F32 => ctx.unify(e0.tv, F32),
                                F64 => ctx.unify(e0.tv, F64),
                                _ => {}
                            }
                        }
                    }
                    Equ | Neq | Gt | Lt | Geq | Leq => {
                        ctx.unify(e0.tv, e1.tv);
                        ctx.unify(self.tv, Bool);
                    }
                    Or | And | Xor => {
                        ctx.unify(self.tv, e0.tv);
                        ctx.unify(self.tv, e1.tv);
                        ctx.unify(self.tv, Bool);
                    }
                    Band | Bor | Bxor => {
                        ctx.unify(e0.tv, e1.tv);
                        ctx.unify(self.tv, e1.tv);
                    }
                    Pipe => unreachable!(),
                    By => ctx.unify(self.tv, TypeKind::By(e0.tv, e1.tv)),
                    Mut => {
                        ctx.unify(self.tv, Unit);
                        ctx.unify(e0.tv, e1.tv);
                    }
                    Seq => ctx.unify(self.tv, e1.tv),
                    In => ctx.unify(self.tv, Bool),
                    NotIn => ctx.unify(self.tv, Bool),
                    BinOpKind::Err => {}
                }
            }
            ExprKind::UnOp(op, e0) => match &op.kind {
                UnOpKind::Boxed => {
                    e0.constrain(ctx);
                    ctx.unify(self.tv, TypeKind::Boxed(e0.tv))
                }
                UnOpKind::Not => {
                    e0.constrain(ctx);
                    ctx.unify(self.tv, e0.tv);
                    ctx.unify(e0.tv, Bool);
                }
                UnOpKind::Neg => ctx.unify(self.tv, e0.tv),
                _ => unreachable!(),
            },
            ExprKind::Call(e, es) => {
                e.constrain(ctx);
                es.constrain(ctx);
                let tvs = es.iter().map(|e| e.tv).collect();
                ctx.unify(e.tv, TypeKind::Fun(tvs, self.tv));
            }
            ExprKind::Select(e0, es) => match es.as_slice() {
                [e1] => {
                    e0.constrain(ctx);
                    e1.constrain(ctx);
                    ctx.unify(e0.tv, TypeKind::Map(e1.tv, self.tv));
                }
                _ => todo!("Probably support arrow-tables?"),
            },
            ExprKind::Project(e, i) => {
                e.constrain(ctx);
                if let TypeKind::Tuple(tvs) = ctx.info.types.resolve(e.tv).kind {
                    if let Some(tv) = tvs.get(i.id) {
                        ctx.unify(self.tv, *tv);
                    } else {
                        ctx.info
                            .diags
                            .intern(Error::OutOfBoundsProject { loc: self.loc })
                    }
                }
            }
            ExprKind::Access(e, x) => {
                e.constrain(ctx);
                if let TypeKind::Struct(fs) = ctx.info.types.resolve(e.tv).kind {
                    if let Some(tv) = fs.get(x) {
                        ctx.unify(self.tv, *tv);
                    } else {
                        ctx.info
                            .diags
                            .intern(Error::FieldNotFound { loc: self.loc })
                    }
                }
            }
            ExprKind::Emit(e) => {
                ctx.unify(self.tv, Unit);
                e.constrain(ctx);
            }
            ExprKind::Log(e) => {
                ctx.unify(self.tv, Unit);
                e.constrain(ctx);
            }
            ExprKind::If(e0, e1, e2) => {
                ctx.unify(e0.tv, Bool);
                ctx.unify(e1.tv, e2.tv);
                ctx.unify(e1.tv, self.tv);
                e0.constrain(ctx);
                e1.constrain(ctx);
                e2.constrain(ctx);
            }
            ExprKind::Loop(_) => todo!(),
            ExprKind::Break => ctx.unify(self.tv, Unit),
            ExprKind::Return(_) => ctx.unify(self.tv, Unit),
            ExprKind::Empty => {}
            ExprKind::Todo => {}
            ExprKind::Add(e0, e1) => {
                e0.constrain(ctx);
                e1.constrain(ctx);
                ctx.unify(e0.tv, hir::TypeKind::Set(e1.tv));
                ctx.unify(self.tv, Unit);
            }
            ExprKind::Del(e0, e1) => {
                e0.constrain(ctx);
                e1.constrain(ctx);
                match ctx.info.types.resolve(e0.tv).kind {
                    hir::TypeKind::Map(tv, _) => ctx.unify(tv, e1.tv),
                    hir::TypeKind::Set(tv) => ctx.unify(tv, e1.tv),
                    hir::TypeKind::Unknown => unknown_err(ctx),
                    _ => ctx
                        .info
                        .diags
                        .intern(Error::ExpectedSelectableType { loc: ctx.loc }),
                }
                ctx.unify(self.tv, Unit);
            }
            ExprKind::Err => {}
        }
        ctx.loc = loc;
    }
}

fn unknown_err(ctx: &mut Context<'_>) {
    ctx.info
        .diags
        .intern(Error::TypeMustBeKnownAtThisPoint { loc: ctx.loc });
}

impl Constrain<'_> for Vec<Expr> {
    fn constrain(&self, ctx: &mut Context<'_>) {
        self.iter().for_each(|e| e.constrain(ctx))
    }
}

impl<T: Eq> Constrain<'_> for VecMap<T, Expr> {
    fn constrain(&self, ctx: &mut Context<'_>) {
        self.values().for_each(|e| e.constrain(ctx))
    }
}
