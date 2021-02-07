/// Module for lowering call-expressions.
mod call;
/// Module for lowering call-expressions.
mod cases;
/// Module for lowering if-let-expressions patterns.
mod if_let;
/// Module for lowering pure lambdas into first-class functions.
mod lambda;
/// Module for lowering let-expressions patterns.
mod let_in;
/// Module for lifting expressions into functions.
mod lift;
/// Module for lowering nominal types.
mod nominal;
/// Module for lowering on clauses.
mod on;
/// Module for lowering path expressions.
mod path;
/// Module for lowering patterns.
mod pattern;
/// Module for lowering ports.
mod ports;
/// Module for lowering names and paths of the AST.
mod resolve;
/// Module for lowering common types.
mod utils;

use crate::compiler::ast;
use crate::compiler::hir::{self, Name, Path};
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;
use crate::compiler::shared::{Lower, Map, New, VecMap};

impl ast::AST {
    pub(crate) fn lower(&self, info: &mut info::Info) -> hir::HIR {
        let res = &mut resolve::Resolver::from(self, info);
        let mut hir = hir::HIR::default();
        let ctx = &mut Context::new(res, self, &mut hir, info);
        for (path_id, module) in &self.modules {
            ctx.res.path_id = *path_id;
            module.lower(ctx);
        }
        hir
    }
}

#[derive(New, Debug)]
pub(crate) struct Context<'i> {
    pub(crate) res: &'i mut resolve::Resolver,
    pub(crate) ast: &'i ast::AST,
    pub(crate) hir: &'i mut hir::HIR,
    pub(crate) info: &'i mut info::Info,
}

/// Resolve a module.
impl Lower<(), Context<'_>> for ast::Module {
    fn lower(&self, ctx: &mut Context) {
        for item in &self.items {
            if let Some(x) = item.lower(ctx) {
                ctx.hir.items.push(x);
            }
        }
    }
}

/// Resolve an item.
impl Lower<Option<Path>, Context<'_>> for ast::Item {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> Option<Path> {
        let (path, kind) = match &self.kind {
            ast::ItemKind::Task(item)  => item.lower(ctx),
            ast::ItemKind::Fun(item)   => item.lower(ctx),
            ast::ItemKind::Alias(item) => item.lower(ctx),
            ast::ItemKind::Enum(item)  => item.lower(ctx),
            ast::ItemKind::Use(item)   => None?,
            ast::ItemKind::Err         => None?,
            ast::ItemKind::Extern(_)   => todo!(),
        };
        let item = hir::Item::new(kind, self.loc.into());
        ctx.hir.defs.insert(path, item);
        path.into()
    }
}

/// Resolve a task item.
impl Lower<Option<Path>, Context<'_>> for ast::TaskItem {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> Option<Path> {
        let (path, kind) = match &self.kind {
            ast::TaskItemKind::Fun(item)   => item.lower(ctx),
            ast::TaskItemKind::Alias(item) => item.lower(ctx),
            ast::TaskItemKind::Enum(item)  => item.lower(ctx),
            ast::TaskItemKind::On(_)       => None?,
            ast::TaskItemKind::Use(_)      => None?,
            ast::TaskItemKind::State(_)    => None?,
            ast::TaskItemKind::Err         => None?,
        };
        let item = hir::Item::new(kind, self.loc.into());
        ctx.hir.defs.insert(path, item);
        path.into()
    }
}

/// Resolve a task state variable.
impl Lower<Option<(Name, hir::Expr)>, Context<'_>> for ast::TaskItem {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> Option<(Name, hir::Expr)> {
        let item = match &self.kind {
            ast::TaskItemKind::Fun(item)   => None?,
            ast::TaskItemKind::Alias(item) => None?,
            ast::TaskItemKind::Enum(item)  => None?,
            ast::TaskItemKind::On(_)       => None?,
            ast::TaskItemKind::Use(_)      => None?,
            ast::TaskItemKind::State(item) => (item.name, item.expr.lower(ctx)),
            ast::TaskItemKind::Err         => None?,
        };
        item.into()
    }
}

/// Resolve a task.
impl Lower<(Path, hir::ItemKind), Context<'_>> for ast::Task {
    fn lower(&self, ctx: &mut Context) -> (Path, hir::ItemKind) {
        tracing::trace!("Lowering Task");
        ctx.res.push_namespace(self.name, ctx.info);
        let mut task_path = ctx.res.path_id;
        let (mut task_params, cases) = pattern::lower_params(&self.params, ctx);
        // If the task's parameter patterns are nested, a function must be created to
        // flatten them. The function is then used in-place of the task.
        if !cases.is_empty() {
            let fun_params = task_params;
            task_params = pattern::params(&self.params, ctx);
            let task_args = pattern::params_to_args(&task_params);

            let fun_path = task_path.into();
            let pathbuf = *ctx.info.paths.resolve(task_path);
            let task_name = ctx.info.names.fresh_with_base(pathbuf.name);
            task_path = ctx
                .info
                .paths
                .intern_child(pathbuf.pred.unwrap(), task_name);

            let ttv = ctx.info.types.fresh();
            let rtv = ctx.info.types.fresh();
            let ftv = ctx.info.types.fresh();

            let call = hir::Expr::syn(
                hir::ExprKind::Call(
                    hir::Expr::syn(hir::ExprKind::Item(task_path.into()), ttv).into(),
                    task_args,
                ),
                rtv,
            );
            let body = pattern::fold_cases(call, None, cases);
            let fun = hir::Fun::new(self.name, fun_params, body, ftv, rtv);
            let item = hir::Item::syn(hir::ItemKind::Fun(fun));
            ctx.hir.defs.insert(fun_path, item);
            ctx.hir.items.push(fun_path);
        }
        // NOTE: These names need to match those which are declared
        let ihub = self.ihub.lower("Source", ctx);
        let ohub = self.ohub.lower("Sink", ctx);

        // TODO: Handle error when there is no handler
        let on = self
            .items
            .iter()
            .filter_map(|item| {
                if let ast::TaskItemKind::On(item) = &item.kind {
                    Some(item)
                } else {
                    None
                }
            })
            .next()
            .unwrap()
            .lower(ctx);
        let mut items = self
            .items
            .iter()
            .filter_map(|item| item.lower(ctx))
            .collect::<Vec<Path>>();
        let tv = ctx.info.types.fresh();
        ctx.res.pop_namespace(ctx.info);
        let task = hir::Task {
            name: self.name,
            tv,
            ihub,
            ohub,
            params: task_params,
            on,
            items,
        };
        (task_path.into(), hir::ItemKind::Task(task))
    }
}

impl ast::Hub {
    fn lower(&self, name: &str, ctx: &mut Context<'_>) -> hir::Hub {
        tracing::trace!("Lowering Hub");
        let kind = match &self.kind {
            ast::HubKind::Tagged(vs) => {
                // Construct enum for ports
                let task_path = ctx.res.path_id;
                let hub_name = ctx.info.names.intern(name).into();
                let hub_path: Path = ctx.info.paths.intern_child(task_path, hub_name).into();
                ctx.res.path_id = hub_path.id;
                let ports = vs.iter().map(|v| v.lower(ctx)).collect::<Vec<_>>();
                let hub_item = hir::Item::syn(hir::ItemKind::Enum(hir::Enum::new(hub_name, ports)));
                ctx.hir.defs.insert(hub_path, hub_item);
                ctx.res.path_id = task_path;
                hir::HubKind::Tagged(hub_path)
            }
            ast::HubKind::Single(ty) => hir::HubKind::Single(ty.lower(ctx)),
        };
        hir::Hub {
            tv: ctx.info.types.fresh(),
            kind,
            loc: self.loc,
        }
    }
}

impl Lower<(Path, hir::ItemKind), Context<'_>> for ast::Fun {
    fn lower(&self, ctx: &mut Context) -> (Path, hir::ItemKind) {
        tracing::trace!("Lowering Fun");
        ctx.res.stack.push_frame();
        let path = ctx.info.paths.intern_child(ctx.res.path_id, self.name);
        let (ps, cases) = pattern::lower_params(&self.params, ctx);
        let e = self.body.lower(ctx);
        let e = pattern::fold_cases(e, None, cases);
        let rtv = self
            .return_ty
            .as_ref()
            .map(|ty| ty.lower(ctx))
            .unwrap_or_else(|| ctx.info.types.intern(hir::ScalarKind::Unit));
        let tv = ctx
            .info
            .types
            .intern(hir::TypeKind::Fun(ps.iter().map(|p| p.tv).collect(), rtv));
        ctx.res.stack.pop_frame();
        let item = hir::Fun::new(self.name, ps, e, tv, rtv);
        (path.into(), hir::ItemKind::Fun(item))
    }
}

/// For now, assume there is a single-case
impl Lower<hir::On, Context<'_>> for ast::On {
    fn lower(&self, ctx: &mut Context) -> hir::On {
        ctx.res.stack.push_scope();
        let mut iter = self.cases.iter();
        let case = iter.next().unwrap();
        let (param, cases) = pattern::lower_pat(&case.pat, ctx);
        for case in cases.iter() {
            tracing::debug!("{}", case.debug(ctx.info, ctx.hir));
        }
        let body = case.body.lower(ctx);
        let tv = ctx.info.types.fresh();
        let else_branch = hir::Expr::syn(hir::ExprKind::Todo, tv);
        let body = pattern::fold_cases(body, Some(else_branch), cases);
        ctx.res.stack.pop_scope();
        hir::On::syn(param, body)
    }
}

impl Lower<(Path, hir::ItemKind), Context<'_>> for ast::Alias {
    fn lower(&self, ctx: &mut Context) -> (Path, hir::ItemKind) {
        let path = ctx.info.paths.intern_child(ctx.res.path_id, self.name);
        let tv = self.ty.lower(ctx);
        let item = hir::Alias::new(self.name, tv);
        (path.into(), hir::ItemKind::Alias(item))
    }
}

impl Lower<(Path, hir::ItemKind), Context<'_>> for ast::Enum {
    fn lower(&self, ctx: &mut Context) -> (Path, hir::ItemKind) {
        let path = ctx.res.path_id;
        let enum_path = ctx.info.paths.intern_child(ctx.res.path_id, self.name);
        ctx.res.path_id = enum_path;
        let variants = self.variants.lower(ctx);
        ctx.res.path_id = path;
        let item = hir::Enum::new(self.name, variants);
        (enum_path.into(), hir::ItemKind::Enum(item))
    }
}

impl Lower<Path, Context<'_>> for ast::Variant {
    fn lower(&self, ctx: &mut Context) -> Path {
        let path: Path = ctx
            .info
            .paths
            .intern_child(ctx.res.path_id, self.name)
            .into();
        let item = hir::Variant::new(
            self.name,
            self.ty
                .as_ref()
                .map(|ty| ty.lower(ctx))
                .unwrap_or_else(|| ctx.info.types.intern(hir::ScalarKind::Unit)),
            self.loc.into(),
        );
        ctx.hir
            .defs
            .insert(path, hir::Item::new(hir::ItemKind::Variant(item), self.loc));
        path
    }
}

impl Lower<hir::Expr, Context<'_>> for ast::Expr {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> hir::Expr {
        let kind = match ctx.ast.exprs.resolve(self.id) {
            ast::ExprKind::Path(x)              => path::lower_path_expr(x, self.loc, ctx),
            ast::ExprKind::Lit(kind)            => hir::ExprKind::Lit(kind.lower(ctx)),
            ast::ExprKind::Array(es)            => hir::ExprKind::Array(es.lower(ctx)),
            ast::ExprKind::Struct(fs)           => hir::ExprKind::Struct(fs.lower(ctx)),
            ast::ExprKind::Tuple(es)            => hir::ExprKind::Tuple(es.lower(ctx)),
            ast::ExprKind::Lambda(ps, e)        => lambda::lower(ps, e, self.loc, ctx),
            ast::ExprKind::Call(e, es)          => call::lower(e, es, ctx),
            ast::ExprKind::UnOp(op, e)          => hir::ExprKind::UnOp(op.lower(ctx), e.lower(ctx).into()),
            ast::ExprKind::BinOp(e0, op, e1)    => hir::ExprKind::BinOp(e0.lower(ctx).into(), op.lower(ctx), e1.lower(ctx).into()),
            ast::ExprKind::If(e0, e1, e2)       => hir::ExprKind::If(e0.lower(ctx).into(), e1.scoped(ctx).into(), e2.scoped(ctx).into()),
            ast::ExprKind::IfLet(p, e0, e1, e2) => return if_let::lower(p, e0, e1, e2, ctx),
            ast::ExprKind::Let(p, e0, e1)       => return let_in::lower(p, e0, e1, ctx),
            ast::ExprKind::Emit(e)              => hir::ExprKind::Emit(e.lower(ctx).into()),
            ast::ExprKind::Unwrap(x, e)         => path::lower_unwrap(x, e, ctx),
            ast::ExprKind::Enwrap(x, e)         => path::lower_enwrap(x, e, ctx),
            ast::ExprKind::Is(x, e)             => path::lower_is(x, e, ctx),
            ast::ExprKind::Log(e)               => hir::ExprKind::Log(e.lower(ctx).into()),
            ast::ExprKind::For(p, e0, e1)       => todo!(),
            ast::ExprKind::Match(e, cs)         => todo!(), //refutable::lower_cases(cs, ctx),
            ast::ExprKind::Loop(e0)             => hir::ExprKind::Loop(e0.scoped(ctx).into()),
            ast::ExprKind::Break                => hir::ExprKind::Break,
            ast::ExprKind::Return(e)            => {
                let e = e
                    .as_ref()
                    .map(|e| e.lower(ctx))
                    .unwrap_or_else(||
                        hir::Expr::syn(
                            hir::ExprKind::Lit(hir::LitKind::Unit),
                            ctx.info.types.intern(hir::ScalarKind::Unit),
                        ));
                hir::ExprKind::Return(e.into())
            },
            ast::ExprKind::Cast(e, ty)       => {
                let mut e = e.lower(ctx);
                e.tv = ty.lower(ctx);
                return e;
            }
            ast::ExprKind::Reduce(p, e, r)   => todo!(),
            ast::ExprKind::Access(e, f)      => hir::ExprKind::Access(e.lower(ctx).into(), *f),
            ast::ExprKind::Project(e, i)     => hir::ExprKind::Project(e.lower(ctx).into(), *i),
            ast::ExprKind::Todo              => hir::ExprKind::Todo,
            ast::ExprKind::Err               => hir::ExprKind::Err,
        };
        hir::Expr::new(kind, ctx.info.types.fresh(), self.loc.into())
    }
}

impl ast::Expr {
    fn scoped(&self, ctx: &mut Context) -> hir::Expr {
        ctx.res.stack.push_scope();
        let expr = self.lower(ctx);
        ctx.res.stack.pop_scope();
        expr
    }
}

impl Lower<hir::UnOp, Context<'_>> for ast::UnOp {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> hir::UnOp {
        let kind = match &self.kind {
            ast::UnOpKind::Not        => hir::UnOpKind::Not,
            ast::UnOpKind::Neg        => hir::UnOpKind::Neg,
            ast::UnOpKind::Err        => hir::UnOpKind::Err,
        };
        hir::UnOp::new(kind, self.loc.into())
    }
}

impl Lower<hir::BinOp, Context<'_>> for ast::BinOp {
    #[rustfmt::skip]
    fn lower(&self, _: &mut Context) -> hir::BinOp {
        hir::BinOp::new(self.kind.clone(), self.loc.into())
    }
}

/// Lower the fields of a struct.
/// NB: Records an error if two fields overlap.
impl<'i, A, B> Lower<VecMap<Name, B>, Context<'i>> for Vec<ast::Field<A>>
where
    A: Lower<B, Context<'i>> + std::fmt::Debug,
{
    fn lower(&self, ctx: &mut Context<'i>) -> VecMap<Name, B> {
        let mut map: VecMap<Name, B> = VecMap::new();
        use crate::compiler::shared::Entry;
        for f in self {
            if let Entry::Vacant(entry) = map.entry(f.name) {
                entry.insert(f.val.lower(ctx));
            } else {
                ctx.info.diags.intern(Error::FieldClash { name: f.name });
            }
        }
        map
    }
}

impl Lower<hir::LitKind, Context<'_>> for ast::LitKind {
    #[rustfmt::skip]
    fn lower(&self, _: &mut Context) -> hir::LitKind {
        self.clone()
    }
}

impl Lower<TypeId, Context<'_>> for ast::Type {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context) -> TypeId {
        let kind = match &self.kind {
            ast::TypeKind::Nominal(path) => nominal::desugar(path, ctx),
            ast::TypeKind::Scalar(kind)  => hir::TypeKind::Scalar(kind.clone()),
            ast::TypeKind::Optional(t)   => hir::TypeKind::Optional(t.lower(ctx)),
            ast::TypeKind::Stream(t)     => hir::TypeKind::Stream(t.lower(ctx)),
            ast::TypeKind::Set(t)        => hir::TypeKind::Set(t.lower(ctx)),
            ast::TypeKind::Vector(t)     => hir::TypeKind::Vector(t.lower(ctx)),
            ast::TypeKind::Array(t, s)   => hir::TypeKind::Array(t.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh()), s.lower(ctx)),
            ast::TypeKind::Fun(ts, t)    => hir::TypeKind::Fun(ts.lower(ctx), t.lower(ctx)),
            ast::TypeKind::Tuple(ts)     => hir::TypeKind::Tuple(ts.lower(ctx)),
            ast::TypeKind::Struct(fs)    => hir::TypeKind::Struct(fs.lower(ctx)),
            ast::TypeKind::Map(t0, t1)   => hir::TypeKind::Map(t0.lower(ctx), t1.lower(ctx)),
            ast::TypeKind::Err           => hir::TypeKind::Err,
        };
        ctx.info.types.intern(kind)
    }
}

impl Lower<hir::Shape, Context<'_>> for ast::Shape {
    fn lower(&self, ctx: &mut Context) -> hir::Shape {
        hir::Shape {
            dims: self.dims.lower(ctx),
        }
    }
}

impl Lower<hir::Dim, Context<'_>> for ast::Dim {
    fn lower(&self, ctx: &mut Context) -> hir::Dim {
        let kind = match &self.kind {
            ast::DimKind::Var(x) => hir::DimKind::Var(*x),
            ast::DimKind::Val(v) => hir::DimKind::Val(*v),
            ast::DimKind::Op(d1, op, d2) => {
                hir::DimKind::Op(d1.lower(ctx).into(), op.lower(ctx), d2.lower(ctx).into())
            }
            ast::DimKind::Err => hir::DimKind::Err,
        };
        hir::Dim { kind }
    }
}

impl Lower<hir::DimOp, Context<'_>> for ast::DimOp {
    fn lower(&self, _: &mut Context) -> hir::DimOp {
        let kind = match &self.kind {
            ast::DimOpKind::Add => hir::DimOpKind::Add,
            ast::DimOpKind::Sub => hir::DimOpKind::Sub,
            ast::DimOpKind::Mul => hir::DimOpKind::Mul,
            ast::DimOpKind::Div => hir::DimOpKind::Div,
        };
        hir::DimOp { kind }
    }
}
