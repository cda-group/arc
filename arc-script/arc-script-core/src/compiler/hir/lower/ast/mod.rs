mod lowerings {
    pub(crate) use super::Context;
    /// Module for lowering call-expressions.
    pub(crate) mod call;
    /// Module for lowering call-expressions.
    //     pub(crate) mod cases;
    /// Module for lowering if-let-expressions patterns.
    pub(crate) mod if_let;
    /// Module for lowering pure lambdas into first-class functions.
    pub(crate) mod lambda;
    /// Module for lowering let-expressions patterns.
    pub(crate) mod let_in;
    /// Module for lifting expressions into functions.
    pub(crate) mod lift;
    /// Module for lowering nominal types.
    pub(crate) mod nominal;
    /// Module for lowering path expressions.
    pub(crate) mod path;
    /// Module for lowering patterns.
    pub(crate) mod pattern;
    /// Module for lowering binop-expressions.
    pub(crate) mod ops;
}

/// Module for lowering names and paths of the AST.
mod resolve;

use lowerings::*;

use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::FunKind::Global;
use crate::compiler::hir::Name;
use crate::compiler::hir::Path;
use crate::compiler::hir::Vis;
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::New;
use arc_script_core_shared::VecMap;

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
    fn lower(&self, ctx: &mut Context<'_>) {
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
    fn lower(&self, ctx: &mut Context<'_>) -> Option<Path> {
        match &self.kind {
            ast::ItemKind::Task(item)   => item.lower(ctx),
            ast::ItemKind::Fun(item)    => {
                ctx.res.stack.push_frame();
                let item = item.lower(ctx, hir::FunKind::Global);
                ctx.res.stack.pop_frame();
                item
            },
            ast::ItemKind::Alias(item)  => item.lower(ctx),
            ast::ItemKind::Enum(item)   => item.lower(ctx),
            ast::ItemKind::Extern(item) => item.lower(ctx, hir::FunKind::Global),
            ast::ItemKind::Use(_)       => None?,
            ast::ItemKind::Err          => None?,
        }
        .map(|(path, kind)| {
            let item = hir::Item::new(kind, self.loc);
            ctx.hir.defs.insert(path, item);
            path
        })
    }
}

/// Resolve a task item.
impl Lower<Option<Path>, Context<'_>> for ast::TaskItem {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) -> Option<Path> {
        match &self.kind {
            ast::TaskItemKind::Fun(item)    => {
                // NOTE: Methods capture their task-environment
                ctx.res.stack.push_scope();
                let item = item.lower(ctx, hir::FunKind::Method);
                ctx.res.stack.pop_scope();
                item
            },
            ast::TaskItemKind::Extern(item) => item.lower(ctx, hir::FunKind::Method),
            ast::TaskItemKind::Alias(item)  => item.lower(ctx),
            ast::TaskItemKind::Enum(item)   => item.lower(ctx),
            ast::TaskItemKind::Port(_)      => None?,
            ast::TaskItemKind::On(_)        => None?,
            ast::TaskItemKind::Use(_)       => None?,
            ast::TaskItemKind::State(_)     => None?,
            ast::TaskItemKind::Err          => None?,
        }
        .map(|(path, kind)| {
            let item = hir::Item::new(kind, self.loc);
            ctx.hir.defs.insert(path, item);
            path
        })
    }
}

impl Lower<hir::State, Context<'_>> for ast::State {
    fn lower(&self, ctx: &mut Context<'_>) -> hir::State {
        let (mut param, cases) = pattern::lower_pat(&self.param.pat, hir::VarKind::State, ctx);
        param.tv = self
            .param
            .ty
            .lower(ctx)
            .unwrap_or_else(|| ctx.info.types.fresh());
        let init = self.expr.lower(ctx);
        hir::State { param, init }
    }
}

/// Resolve a task.
impl Lower<Option<(Path, hir::ItemKind)>, Context<'_>> for ast::Task {
    fn lower(&self, ctx: &mut Context<'_>) -> Option<(Path, hir::ItemKind)> {
        tracing::trace!("Lowering Task");
        ctx.res.push_namespace(self.name, ctx.info);
        let mut task_path: Path = ctx.res.path_id.into();
        let (mut task_params, cases) =
            pattern::lower_params(&self.params, hir::VarKind::Member, ctx);
        let states: Vec<hir::State> = self
            .items
            .iter()
            .filter_map(|item| {
                map!(&item.kind, ast::TaskItemKind::State).map(|item| item.lower(ctx))
            })
            .collect();
        // If the task's parameter patterns are nested, a function must be created to
        // flatten them. The function is then used in-place of the task.
        if !cases.is_empty() {
            let fun_params =
                std::mem::replace(&mut task_params, pattern::params(&self.params, ctx));
            let task_args = pattern::params_to_args(&task_params, hir::VarKind::Local);

            let path = *ctx.info.paths.resolve(task_path.id);
            let task_name = ctx.info.names.fresh_with_base(path.name);
            let fun_path = std::mem::replace(
                &mut task_path,
                ctx.info
                    .paths
                    .intern_child(path.pred.unwrap(), task_name)
                    .into(),
            );

            let ttv = ctx.info.types.fresh();
            let rtv = ctx.info.types.fresh();
            let ftv = ctx.info.types.fresh();

            let call = hir::Expr::syn(
                hir::ExprKind::Call(
                    hir::Expr::syn(hir::ExprKind::Item(task_path), ttv).into(),
                    task_args,
                ),
                rtv,
            );
            let body = pattern::fold_cases(call, None, cases);
            let fun = hir::Fun::new(Global, fun_path, fun_params, None, body, ftv, rtv);
            let item = hir::Item::syn(hir::ItemKind::Fun(fun));
            ctx.hir.defs.insert(fun_path, item);
            ctx.hir.items.push(fun_path);
        }
        // NOTE: These names need to match those which are declared
        let ihub = self
            .ihub
            .lower(ctx.info.names.common.source.into(), &self.items, ctx);
        let ohub = self
            .ohub
            .lower(ctx.info.names.common.sink.into(), &self.items, ctx);

        let on = self
            .items
            .iter()
            .find_map(|item| map!(&item.kind, ast::TaskItemKind::On))
            .map(|item| item.lower(ctx));
        let items = self
            .items
            .iter()
            .filter_map(|item| item.lower(ctx))
            .collect::<Vec<Path>>();
        let tv = ctx.info.types.fresh();
        ctx.res.pop_namespace(ctx.info);
        let task = hir::Task {
            path: task_path,
            tv,
            ihub,
            ohub,
            params: task_params,
            states,
            on,
            items,
        };
        (task_path, hir::ItemKind::Task(task)).into()
    }
}

impl ast::Hub {
    fn lower(
        &self,
        hub_name: Name,
        task_items: &[ast::TaskItem],
        ctx: &mut Context<'_>,
    ) -> hir::Hub {
        tracing::trace!("Lowering Hub");
        let kind = match &self.kind {
            ast::HubKind::Tagged(ports) => {
                // Construct enum for ports
                let task_path = ctx.res.path_id;
                let hub_path: Path = ctx.info.paths.intern_child(task_path, hub_name).into();
                let mut ports = ports
                    .iter()
                    .map(|v| v.lower(hub_path.id, ctx))
                    .collect::<Vec<_>>();
                ports.extend(task_items.iter().filter_map(|item| {
                    map!(&item.kind, ast::TaskItemKind::Port)
                        .map(|v| v.lower(hub_path.id, item.loc, ctx))
                }));
                let hub_item = hir::Item::syn(hir::ItemKind::Enum(hir::Enum::new(hub_path, ports)));
                ctx.hir.defs.insert(hub_path, hub_item);
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

impl ast::Fun {
    fn lower(&self, ctx: &mut Context<'_>, kind: hir::FunKind) -> Option<(Path, hir::ItemKind)> {
        tracing::trace!("Lowering Fun");
        let path = ctx
            .info
            .paths
            .intern_child(ctx.res.path_id, self.name)
            .into();
        let (params, cases) = pattern::lower_params(&self.params, hir::VarKind::Local, ctx);
        let e = self.body.lower(ctx);
        let e = pattern::fold_cases(e, None, cases);
        let (channels, e) = if let Some(channels) = &self.channels {
            let (channels, cases) = pattern::lower_params(channels, hir::VarKind::Local, ctx);
            let e = pattern::fold_cases(e, None, cases);
            (Some(channels), e)
        } else {
            (None, e)
        };
        let rtv = self
            .return_ty
            .as_ref()
            .map(|ty| ty.lower(ctx))
            .unwrap_or_else(|| ctx.info.types.intern(hir::ScalarKind::Unit));
        let tv = ctx.info.types.intern(hir::TypeKind::Fun(
            params.iter().map(|p| p.tv).collect(),
            rtv,
        ));
        let item = hir::Fun::new(kind, path, params, channels, e, tv, rtv);
        (path, hir::ItemKind::Fun(item)).into()
    }
}

impl ast::Extern {
    fn lower(&self, ctx: &mut Context<'_>, kind: hir::FunKind) -> Option<(Path, hir::ItemKind)> {
        tracing::trace!("Lowering Extern");
        let no_patterns = self
            .params
            .iter()
            .all(|p| matches!(p.pat.kind, ast::PatKind::Var(_)));
        if no_patterns {
            let params = self
                .params
                .iter()
                .filter_map(|p| {
                    map!(&p.pat.kind, ast::PatKind::Var).map(|x| {
                        let tv = p.ty.as_ref().unwrap().lower(ctx);
                        hir::Param::new(hir::ParamKind::Var(*x), tv, p.loc)
                    })
                })
                .collect::<Vec<_>>();
            let path = ctx
                .info
                .paths
                .intern_child(ctx.res.path_id, self.name)
                .into();
            let rtv = self.return_ty.lower(ctx);
            let tv = ctx.info.types.intern(hir::TypeKind::Fun(
                params.iter().map(|p| p.tv).collect(),
                rtv,
            ));
            let item = hir::Extern::new(kind, path, params, tv, rtv);
            (path, hir::ItemKind::Extern(item)).into()
        } else {
            ctx.info
                .diags
                .intern(Error::PatternInExternFun { loc: self.name.loc });
            None
        }
    }
}

/// For now, assume there is a single-case
impl Lower<hir::On, Context<'_>> for ast::On {
    fn lower(&self, ctx: &mut Context<'_>) -> hir::On {
        ctx.res.stack.push_scope();
        let mut iter = self.cases.iter();
        let case = iter.next().unwrap();
        let (param, cases) = pattern::lower_pat(&case.pat, hir::VarKind::Local, ctx);
        for case in cases.iter() {
            tracing::debug!("{}", case.debug(ctx.info, ctx.hir));
        }
        let body = case.body.lower(ctx);
        let tv = ctx.info.types.fresh();
        let else_branch = hir::Expr::syn(hir::ExprKind::Todo, tv);
        let body = pattern::fold_cases(body, Some(&else_branch), cases);
        ctx.res.stack.pop_scope();
        hir::On::syn(param, body)
    }
}

impl Lower<Option<(Path, hir::ItemKind)>, Context<'_>> for ast::Alias {
    fn lower(&self, ctx: &mut Context<'_>) -> Option<(Path, hir::ItemKind)> {
        let path = ctx
            .info
            .paths
            .intern_child(ctx.res.path_id, self.name)
            .into();
        let tv = self.ty.lower(ctx);
        let item = hir::Alias::new(path, tv);
        (path, hir::ItemKind::Alias(item)).into()
    }
}

impl Lower<Option<(Path, hir::ItemKind)>, Context<'_>> for ast::Enum {
    fn lower(&self, ctx: &mut Context<'_>) -> Option<(Path, hir::ItemKind)> {
        let path: Path = ctx
            .info
            .paths
            .intern_child(ctx.res.path_id, self.name)
            .into();
        let variants = self
            .variants
            .iter()
            .map(|v| v.lower(path.id, ctx))
            .collect::<Vec<_>>();
        let item = hir::Enum::new(path, variants);
        (path, hir::ItemKind::Enum(item)).into()
    }
}

impl ast::Variant {
    fn lower(&self, enum_path: hir::PathId, ctx: &mut Context<'_>) -> Path {
        let path: Path = ctx.info.paths.intern_child(enum_path, self.name).into();
        let tv = self
            .ty
            .as_ref()
            .map(|ty| ty.lower(ctx))
            .unwrap_or_else(|| ctx.info.types.intern(hir::ScalarKind::Unit));
        let item = hir::Variant::new(Vis::Public, path, tv, self.loc);
        ctx.hir
            .defs
            .insert(path, hir::Item::new(hir::ItemKind::Variant(item), self.loc));
        path
    }
}

impl ast::InnerPort {
    fn lower(&self, enum_path: hir::PathId, loc: Option<Loc>, ctx: &mut Context<'_>) -> Path {
        let path: Path = ctx.info.paths.intern_child(enum_path, self.name).into();
        let tv = self.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
        let item = hir::Variant::new(Vis::Private, path, tv, loc);
        ctx.hir
            .defs
            .insert(path, hir::Item::new(hir::ItemKind::Variant(item), loc));
        path
    }
}

impl ast::Port {
    fn lower(&self, enum_path: hir::PathId, ctx: &mut Context<'_>) -> Path {
        let path: Path = ctx.info.paths.intern_child(enum_path, self.name).into();
        let item = hir::Variant::new(Vis::Public, path, self.ty.lower(ctx), self.loc);
        ctx.hir
            .defs
            .insert(path, hir::Item::new(hir::ItemKind::Variant(item), self.loc));
        path
    }
}

impl Lower<hir::Expr, Context<'_>> for ast::Expr {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) -> hir::Expr {
        let kind = match ctx.ast.exprs.resolve(self.id) {
            ast::ExprKind::Path(x)              => path::lower_path_expr(x, self.loc, ctx),
            ast::ExprKind::Lit(kind)            => hir::ExprKind::Lit(kind.lower(ctx)),
            ast::ExprKind::Array(es)            => hir::ExprKind::Array(es.as_slice().lower(ctx)),
            ast::ExprKind::Struct(fs)           => hir::ExprKind::Struct(fs.lower(ctx)),
            ast::ExprKind::Tuple(es)            => hir::ExprKind::Tuple(es.as_slice().lower(ctx)),
            ast::ExprKind::Lambda(ps, e)        => lambda::lower(ps, e, self.loc, ctx),
            ast::ExprKind::Call(e, es)          => call::lower(e, es, ctx),
            ast::ExprKind::Select(e, es)        => hir::ExprKind::Select(e.lower(ctx).into(), es.as_slice().lower(ctx)),
            ast::ExprKind::UnOp(op, e) => match op.kind {
                ast::UnOpKind::Add => ops::lower_add(e, self.loc, ctx),
                ast::UnOpKind::Del => ops::lower_del(e, self.loc, ctx),
                _ => hir::ExprKind::UnOp(op.lower(ctx), e.lower(ctx).into())
            },
            ast::ExprKind::BinOp(e0, op, e1) => match op.kind {
                ast::BinOpKind::NotIn => ops::lower_not_in(e0, e1, self.loc, ctx),
                ast::BinOpKind::Pipe  => ops::lower_pipe(e0, e1, self.loc, ctx),
                ast::BinOpKind::After => ops::lower_after(e0, e1, self.loc, ctx),
                _                     => hir::ExprKind::BinOp(e0.lower(ctx).into(), op.lower(ctx), e1.lower(ctx).into())

            }
            ast::ExprKind::If(e0, e1, e2)       => hir::ExprKind::If(e0.lower(ctx).into(), e1.scoped(ctx).into(), e2.scoped(ctx).into()),
            ast::ExprKind::IfLet(p, e0, e1, e2) => return if_let::lower(p, e0, e1, e2, ctx),
            ast::ExprKind::Let(p, e0, e1)       => return let_in::lower(p, e0, e1, ctx),
            ast::ExprKind::Emit(e)              => hir::ExprKind::Emit(e.lower(ctx).into()),
            ast::ExprKind::Unwrap(x, e)         => path::lower_unwrap(x, e, ctx),
            ast::ExprKind::Enwrap(x, e)         => path::lower_enwrap(x, e, ctx),
            ast::ExprKind::Is(x, e)             => path::lower_is(x, e, ctx),
            ast::ExprKind::Log(e)               => hir::ExprKind::Log(e.lower(ctx).into()),
            ast::ExprKind::For(_p, _e0, _e1)    => todo!(),
            ast::ExprKind::Match(_e, _cs)       => todo!(), //refutable::lower_cases(cs, ctx),
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
            ast::ExprKind::Reduce(_p, _e, _r)   => todo!(),
            ast::ExprKind::Access(e, f)      => hir::ExprKind::Access(e.lower(ctx).into(), *f),
            ast::ExprKind::Project(e, i)     => hir::ExprKind::Project(e.lower(ctx).into(), *i),
            ast::ExprKind::Empty             => hir::ExprKind::Empty,
            ast::ExprKind::Todo              => hir::ExprKind::Todo,
            ast::ExprKind::Err               => hir::ExprKind::Err,
        };
        hir::Expr::new(kind, ctx.info.types.fresh(), self.loc)
    }
}

impl ast::Expr {
    fn scoped(&self, ctx: &mut Context<'_>) -> hir::Expr {
        ctx.res.stack.push_scope();
        let expr = self.lower(ctx);
        ctx.res.stack.pop_scope();
        expr
    }
}

impl Lower<hir::UnOp, Context<'_>> for ast::UnOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> hir::UnOp {
        hir::UnOp::new(self.kind.clone(), self.loc)
    }
}

impl Lower<hir::BinOp, Context<'_>> for ast::BinOp {
    fn lower(&self, _: &mut Context<'_>) -> hir::BinOp {
        hir::BinOp::new(self.kind.clone(), self.loc)
    }
}

/// Lower the fields of a struct.
/// NB: Records an error if two fields overlap.
impl<'i, A, B> Lower<VecMap<Name, B>, Context<'i>> for Vec<ast::Field<A>>
where
    A: Lower<B, Context<'i>> + std::fmt::Debug,
{
    fn lower(&self, ctx: &mut Context<'i>) -> VecMap<Name, B> {
        use arc_script_core_shared::VecMapEntry;
        let mut map: VecMap<Name, B> = VecMap::new();
        for f in self {
            if let VecMapEntry::Vacant(entry) = map.entry(f.name) {
                entry.insert(f.val.lower(ctx));
            } else {
                ctx.info.diags.intern(Error::FieldClash { name: f.name });
            }
        }
        map
    }
}

impl Lower<hir::LitKind, Context<'_>> for ast::LitKind {
    fn lower(&self, _: &mut Context<'_>) -> Self {
        self.clone()
    }
}

impl Lower<TypeId, Context<'_>> for ast::Type {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) -> TypeId {
        let kind = match &self.kind {
            ast::TypeKind::Nominal(path) => nominal::desugar(path, ctx),
            ast::TypeKind::Scalar(kind)  => hir::TypeKind::Scalar(*kind),
            ast::TypeKind::Optional(t)   => hir::TypeKind::Optional(t.lower(ctx)),
            ast::TypeKind::Stream(t)     => hir::TypeKind::Stream(t.lower(ctx)),
            ast::TypeKind::Set(t)        => hir::TypeKind::Set(t.lower(ctx)),
            ast::TypeKind::Vector(t)     => hir::TypeKind::Vector(t.lower(ctx)),
            ast::TypeKind::Array(t, s)   => hir::TypeKind::Array(t.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh()), s.lower(ctx)),
            ast::TypeKind::Fun(ts, t)    => hir::TypeKind::Fun(ts.as_slice().lower(ctx), t.lower(ctx)),
            ast::TypeKind::Tuple(ts)     => hir::TypeKind::Tuple(ts.as_slice().lower(ctx)),
            ast::TypeKind::Struct(fs)    => hir::TypeKind::Struct(fs.lower(ctx)),
            ast::TypeKind::Map(t0, t1)   => hir::TypeKind::Map(t0.lower(ctx), t1.lower(ctx)),
            ast::TypeKind::Boxed(ty)     => hir::TypeKind::Boxed(ty.lower(ctx)),
            ast::TypeKind::By(t0, t1)    => hir::TypeKind::By(t0.lower(ctx), t1.lower(ctx)),
            ast::TypeKind::Err           => hir::TypeKind::Err,
        };
        ctx.info.types.intern(kind)
    }
}

impl Lower<hir::Shape, Context<'_>> for ast::Shape {
    fn lower(&self, ctx: &mut Context<'_>) -> hir::Shape {
        hir::Shape {
            dims: self.dims.as_slice().lower(ctx),
        }
    }
}

impl Lower<hir::Dim, Context<'_>> for ast::Dim {
    fn lower(&self, ctx: &mut Context<'_>) -> hir::Dim {
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
    fn lower(&self, _: &mut Context<'_>) -> hir::DimOp {
        let kind = match &self.kind {
            ast::DimOpKind::Add => hir::DimOpKind::Add,
            ast::DimOpKind::Sub => hir::DimOpKind::Sub,
            ast::DimOpKind::Mul => hir::DimOpKind::Mul,
            ast::DimOpKind::Div => hir::DimOpKind::Div,
        };
        hir::DimOp { kind }
    }
}
