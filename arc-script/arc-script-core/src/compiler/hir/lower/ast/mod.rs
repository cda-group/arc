/// Module for lowering names and paths of the AST.
pub(crate) mod resolve;
mod utils;
/// Special lowerings.
mod special;

use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::utils::SortFields;
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::get;
use arc_script_core_shared::itertools::Itertools;
use arc_script_core_shared::lower;
use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;
use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;
use arc_script_core_shared::VecMap;

use std::collections::VecDeque;

use tracing::instrument;

impl ast::AST {
    #[instrument(name = "(Lower)", level = "debug", skip(self, res, info))]
    pub(crate) fn lower(&self, res: &mut resolve::Resolver, info: &mut info::Info) -> hir::HIR {
        let mut hir = hir::HIR::default();
        let ctx = &mut Context {
            res,
            ast: self,
            hir: &mut hir,
            info,
            generated_ointerface_interior: None,
            generated_iinterface_interior: None,
            state_enum: None,
        };
        for (path, module) in &self.modules {
            ctx.res.set_path(*path);
            module.lower(ctx);
        }
        hir
    }
}

#[derive(New, Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    pub(crate) res: &'i mut resolve::Resolver,
    pub(crate) ast: &'i ast::AST,
    pub(crate) hir: &'i mut hir::HIR,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut info::Info,
    /// Tasks with untagged ports are lowered into tasks with tagged ports. This is set to
    /// `Some(<path>)` where <path> is a generated output interface enum which allows us to
    /// wrap emit expressions as `<path>::__(<expr>)`.
    pub(crate) generated_ointerface_interior: Option<hir::Path>,
    pub(crate) generated_iinterface_interior: Option<hir::Path>,
    pub(crate) state_enum: Option<hir::Path>,
}

lower! {
    [node, ctx, ast]

    ast::Module => () {
        for item in &node.items {
            if let Some(x) = item.lower(ctx) {
                ctx.hir.namespace.push(x);
            }
        }
    },
    Vec<ast::Item> => Vec<hir::PathId> {
        node.iter().filter_map(|item| item.lower(ctx)).collect()
    },
    ast::Item => Option<hir::PathId> {
        node.kind.lower(ctx).map(|kind| {
            let path = kind.get_path();
            ctx.hir.intern(path, hir::Item::new(kind, node.loc));
            *path
        })
    },
    ast::ItemKind => Option<hir::ItemKind> {
        let kind = match node {
            ast::ItemKind::Task(item)       => item.lower(ctx).into(),
            ast::ItemKind::Fun(item)        => item.lower(ctx).into(),
            ast::ItemKind::TypeAlias(item)  => item.lower(ctx).into(),
            ast::ItemKind::Enum(item)       => item.lower(ctx).into(),
            ast::ItemKind::ExternFun(item)  => item.lower(ctx).into(),
            ast::ItemKind::ExternType(item) => item.lower(ctx).into(),
            ast::ItemKind::Assign(_)        => crate::todo!(),
            ast::ItemKind::Use(_)           => None?,
            ast::ItemKind::Err              => None?,
        };
        Some(kind)
    },
    ast::Assign => () {
        let cases = special::pattern::lower_assign(node, false, hir::ScopeKind::Local, ctx);
        ctx.get_stmts().append(&mut special::pattern::cases_to_stmts(cases));
    },
    ast::Fun => hir::Fun {
        // Push frame to prevent the function from capturing variables in outer scopes
        ctx.res.stack.push_frame();
        let path = ctx.new_path(node.name);
        let (params, cases) = special::pattern::lower_params(&node.params, hir::ScopeKind::Local, ctx);
        let mut block = node.block.lower(ctx);
        block.prepend_stmts(special::pattern::cases_to_stmts(cases));
        let return_t = ctx.new_type_unit_if_none(&node.rt);
        let param_ts = params.iter().map(|p| p.t).collect();
        let fun_t = ctx.types.intern(hir::TypeKind::Fun(param_ts, return_t));
        ctx.res.stack.pop_frame();
        hir::Fun::new(path, hir::FunKind::Method, params, block, fun_t, return_t)
    },
    // Block-expressions need to be flattened. For this reason, the context maintains a list of
    // statements of the current block.
    ast::Block => hir::Block {
        ctx.res.stack.push_scope();
        node.stmts.lower(ctx);
        let v = node.expr.lower(ctx);
        let stmts = ctx.res.stack.pop_scope();
        hir::Block::new(stmts, v, node.loc)
    },
    Vec<ast::Stmt> => () {
        node.iter().for_each(|stmt| stmt.lower(ctx))
    },
    // Statements are SSA:d, so they don't lower into anything on their own
    ast::Stmt => () {
        match &node.kind {
            ast::StmtKind::Empty     => {},
            ast::StmtKind::Assign(a) => { a.lower(ctx); },
            ast::StmtKind::Expr(e)   => { e.lower(ctx); },
        }
    },
    Option<ast::Expr> => hir::Var {
        if let Some(e) = node {
            e.lower(ctx)
        } else {
            ctx.new_expr_unit().into_ssa(ctx)
        }
    },
    ast::ExternType => hir::ExternType {
        ctx.res.push_namespace(node.name, ctx.info);
        let params = node.params.iter().map(|p| {
                let x = get!(ctx.ast.pats.resolve(&p.pat), ast::PatKind::Var(x));
                let t = p.t.as_ref().unwrap().lower(ctx);
                hir::Param::new(hir::ParamKind::Ok(*x), t, p.loc)
            })
            .collect::<Vec<_>>();
        let items = node.items.lower(ctx);
        let path = ctx.res.path.into();
        let t = ctx.types.fresh();
        ctx.res.pop_namespace(ctx.info);
        hir::ExternType::new(path, params, items, t)
    },
    Vec<ast::ExternTypeItem> => Vec<hir::PathId> {
        node.iter().filter_map(|item| item.lower(ctx)).collect()
    },
    ast::ExternTypeItem => Option<hir::PathId> {
        node.kind.lower(ctx).map(|kind| {
            let path = kind.get_path();
            ctx.hir.intern(path, hir::Item::new(kind, node.loc));
            *path
        })
    },
    ast::ExternTypeItemKind => Option<hir::ItemKind> {
        let kind = match node {
            ast::ExternTypeItemKind::ExternMethod(item) => item.lower(ctx).into(),
            ast::ExternTypeItemKind::Err => None?,
        };
        Some(kind)
    },
    ast::ExternFun => hir::ExternFun {
        let path = ctx.new_path(node.name);
        let params = node.params.iter().map(|p| {
                let x = get!(ctx.ast.pats.resolve(&p.pat), ast::PatKind::Var(x));
                let t = p.t.as_ref().unwrap().lower(ctx);
                hir::Param::new(hir::ParamKind::Ok(*x), t, p.loc)
            })
            .collect::<Vec<_>>();
        let rt = node.rt.lower(ctx);
        let ts = params.iter().map(|p| p.t).collect();
        let t = ctx.types.intern(hir::TypeKind::Fun(ts, rt));
        hir::ExternFun::new(path, hir::FunKind::Free, params, t, rt)
    },
    ast::ExternMethod => hir::ExternFun {
        let path = ctx.new_path(node.name);
        let params = node.params.iter().map(|p| {
                let x = get!(ctx.ast.pats.resolve(&p.pat), ast::PatKind::Var(x));
                let t = p.t.as_ref().unwrap().lower(ctx);
                hir::Param::new(hir::ParamKind::Ok(*x), t, p.loc)
            })
            .collect::<Vec<_>>();
        let rt = node.rt.lower(ctx);
        let ts = params.iter().map(|p| p.t).collect();
        let t = ctx.types.intern(hir::TypeKind::Fun(ts, rt));
        hir::ExternFun::new(path, hir::FunKind::Method, params, t, rt)
    },
    ast::Task => hir::Task {
        ctx.res.stack.push_frame();
        let mut path: hir::Path = ctx.res.path.into();
        let (params, cases) = special::pattern::lower_params(&node.params, hir::ScopeKind::Member, ctx);
        assert!(cases.is_empty());
        let iinterface = node.iinterface.lower(ctx.names.common.iinterface, ctx);
        let ointerface = node.ointerface.lower(ctx.names.common.ointerface, ctx);
        if matches!(node.ointerface.kind, ast::InterfaceKind::Single(_)) {
            ctx.generated_iinterface_interior = Some(iinterface.interior);
        }
        if matches!(node.iinterface.kind, ast::InterfaceKind::Single(_)) {
            ctx.generated_ointerface_interior = Some(ointerface.interior);
        }
        let fsm = special::fsm::lower_fsm(&node.block, ctx);
        ctx.generated_ointerface_interior = None;
        ctx.generated_iinterface_interior = None;

        let cons_t = ctx.types.fresh();
        let fun_t = ctx.types.fresh();
        let struct_t = ctx.types.fresh();
        ctx.res.stack.pop_frame();
        let cons_x = path; // TODO
        hir::Task {
            path,
            cons_x,
            cons_t,
            fun_t,
            struct_t,
            params,
            iinterface,
            ointerface,
            fsm
        }
    },
    ast::TypeAlias => hir::TypeAlias {
        let path = ctx.new_path(node.name);
        let t = node.t.lower(ctx);
        hir::TypeAlias::new(path, t)
    },
    ast::Enum => hir::Enum {
        let path = ctx.new_path(node.name);
        let variants = node
            .variants
            .iter()
            .map(|v| {
                let path: hir::Path = ctx.paths.intern_child(path, v.name).into();
                let t = ctx.new_type_unit_if_none(&v.t);
                let item = hir::Item::new(hir::Variant::new(path, t, v.loc).into(), v.loc);
                ctx.hir.intern(path, item);
                path
            })
            .collect::<Vec<_>>();
        hir::Enum::new(path, variants)
    },
    ast::Expr => hir::Var {
        let kind = match ctx.ast.exprs.resolve(node) {
            ast::ExprKind::Return(e)           => hir::ExprKind::Return(e.lower(ctx)),
            ast::ExprKind::Break(e)            => hir::ExprKind::Break(e.lower(ctx)),
            ast::ExprKind::Continue            => hir::ExprKind::Continue,
            ast::ExprKind::Path(x, ts)         => return special::path::lower_expr_path(x, ts, node.loc, ctx),
            ast::ExprKind::Lambda(ps, e)       => special::lambda::lower(ps, e, node.loc, ctx),
            ast::ExprKind::Call(e, es)         => special::call::lower(e, es, ctx),
            ast::ExprKind::Invoke(e, x, es)    => hir::ExprKind::Invoke(e.lower(ctx), *x, es.as_slice().lower(ctx)),
            ast::ExprKind::If(e, b0, b1)       => special::if_else::lower_if(e, b0, b1, ctx),
            ast::ExprKind::IfAssign(a, b0, b1) => return special::if_else::lower_if_assign(a, b0, b1, ctx),
            ast::ExprKind::Lit(kind)           => hir::ExprKind::Lit(kind.lower(ctx)),
            ast::ExprKind::Array(es)           => hir::ExprKind::Array(es.as_slice().lower(ctx)),
            ast::ExprKind::Struct(fs)          => hir::ExprKind::Struct(fs.lower(ctx)),
            ast::ExprKind::Tuple(es)           => hir::ExprKind::Tuple(es.as_slice().lower(ctx)),
            ast::ExprKind::Select(e, es)       => hir::ExprKind::Select(e.lower(ctx), es.as_slice().lower(ctx)),
            ast::ExprKind::UnOp(op, e)         => hir::ExprKind::UnOp(*op, e.lower(ctx)),
            ast::ExprKind::BinOp(e0, op, e1)   => match op.kind {
                ast::BinOpKind::NotIn => special::ops::lower_notin(e0, e1, node.loc, ctx),
                ast::BinOpKind::Pipe  => special::ops::lower_pipe(e0, e1, node.loc, ctx),
                ast::BinOpKind::By    => hir::ExprKind::Struct(special::ops::lower_by(e0, e1, ctx)),
                _                     => hir::ExprKind::BinOp(e0.lower(ctx), op.lower(ctx), e1.lower(ctx))
            }
            ast::ExprKind::Emit(e)             => special::task::lower_emit(e, ctx),
            ast::ExprKind::Unwrap(x, e)        => special::path::lower_unwrap_path(x, e, ctx),
            ast::ExprKind::Enwrap(x, e)        => special::path::lower_enwrap_path(x, e, ctx),
            ast::ExprKind::Is(x, e)            => special::path::lower_is_path(x, e, ctx),
            ast::ExprKind::On(cs)              => special::task::lower_on(cs, ctx),
            ast::ExprKind::Log(e)              => hir::ExprKind::Log(e.lower(ctx)),
            ast::ExprKind::For(_p, _e0, _e1)   => crate::todo!(),
            ast::ExprKind::Match(_e, _cs)      => crate::todo!(), //refutable::lower_cases(cs, ctx),
            ast::ExprKind::Loop(b)             => special::loops::lower_loop(b, ctx),
            ast::ExprKind::Cast(e, ty)         => hir::ExprKind::Cast(e.lower(ctx), ty.lower(ctx)),
            ast::ExprKind::Access(e, f)        => hir::ExprKind::Access(e.lower(ctx), *f),
            ast::ExprKind::Project(e, i)       => hir::ExprKind::Project(e.lower(ctx), *i),
            ast::ExprKind::After(e0, e1)       => hir::ExprKind::After(e0.lower(ctx), e1.lower(ctx)),
            ast::ExprKind::Every(e0, e1)       => hir::ExprKind::Every(e0.lower(ctx), e1.lower(ctx)),
            ast::ExprKind::Err                 => hir::ExprKind::Err,
            ast::ExprKind::Block(b)            => {
                // Flatten block
                let mut b = b.lower(ctx);
                ctx.get_stmts().append(&mut b.stmts);
                return b.var;
            },
        };
        ctx.new_expr_with_loc(kind, node.loc).into_ssa(ctx)
    },
    ast::BinOp => hir::BinOp {
        hir::BinOp::new(node.kind.lower(ctx), node.loc)
    },
    ast::BinOpKind => hir::BinOpKind {
        match node {
            ast::BinOpKind::Add   => hir::BinOpKind::Add,
            ast::BinOpKind::And   => hir::BinOpKind::And,
            ast::BinOpKind::Band  => hir::BinOpKind::Band,
            ast::BinOpKind::Bor   => hir::BinOpKind::Bor,
            ast::BinOpKind::Bxor  => hir::BinOpKind::Bxor,
            ast::BinOpKind::Div   => hir::BinOpKind::Div,
            ast::BinOpKind::Equ   => hir::BinOpKind::Equ,
            ast::BinOpKind::Geq   => hir::BinOpKind::Geq,
            ast::BinOpKind::Gt    => hir::BinOpKind::Gt,
            ast::BinOpKind::Leq   => hir::BinOpKind::Leq,
            ast::BinOpKind::Lt    => hir::BinOpKind::Lt,
            ast::BinOpKind::Mod   => hir::BinOpKind::Mod,
            ast::BinOpKind::Mul   => hir::BinOpKind::Mul,
            ast::BinOpKind::Mut   => hir::BinOpKind::Mut,
            ast::BinOpKind::Neq   => hir::BinOpKind::Neq,
            ast::BinOpKind::Or    => hir::BinOpKind::Or,
            ast::BinOpKind::Pow   => hir::BinOpKind::Pow,
            ast::BinOpKind::Sub   => hir::BinOpKind::Sub,
            ast::BinOpKind::Xor   => hir::BinOpKind::Xor,
            ast::BinOpKind::Err   => hir::BinOpKind::Err,
            ast::BinOpKind::By    => unreachable!(),
            ast::BinOpKind::In    => unreachable!(),
            ast::BinOpKind::NotIn => unreachable!(),
            ast::BinOpKind::Pipe  => unreachable!(),
            ast::BinOpKind::RExc  => unreachable!(),
            ast::BinOpKind::RInc  => unreachable!(),
        }
    },
    ast::LitKind => hir::LitKind {
        node.clone()
    },
    ast::Type => hir::Type {
        let kind = match ctx.ast.types.resolve(*node) {
            ast::TypeKind::Path(x, ts)   => special::path::lower_type_path(x, ts, node.loc, ctx),
            ast::TypeKind::Scalar(kind)  => hir::TypeKind::Scalar(*kind),
            ast::TypeKind::Stream(t)     => hir::TypeKind::Stream(t.lower(ctx)),
            ast::TypeKind::Array(t, s)   => hir::TypeKind::Array(t.lower(ctx).unwrap_or_else(|| ctx.types.fresh()), s.lower(ctx)),
            ast::TypeKind::Fun(ts, t)    => hir::TypeKind::Fun(ts.as_slice().lower(ctx), t.lower(ctx)),
            ast::TypeKind::Tuple(ts)     => hir::TypeKind::Tuple(ts.as_slice().lower(ctx)),
            ast::TypeKind::Struct(fs)    => hir::TypeKind::Struct(fs.lower(ctx).sort_fields(ctx.info)),
            ast::TypeKind::By(t0, t1)    => hir::TypeKind::Struct(special::ops::lower_by(t0, t1, ctx).sort_fields(ctx.info)),
            ast::TypeKind::Err           => hir::TypeKind::Err,
        };
        ctx.info.types.intern(kind)
    },
    ast::Shape => hir::Shape {
        hir::Shape {
            dims: node.dims.as_slice().lower(ctx),
        }
    },
    ast::Dim => hir::Dim {
        let kind = match &node.kind {
            ast::DimKind::Var(x) => hir::DimKind::Var(*x),
            ast::DimKind::Val(v) => hir::DimKind::Val(*v),
            ast::DimKind::Op(d1, op, d2) => {
                hir::DimKind::Op(d1.lower(ctx).into(), op.lower(ctx), d2.lower(ctx).into())
            }
            ast::DimKind::Err => hir::DimKind::Err,
        };
        hir::Dim { kind }
    },
    ast::DimOp => hir::DimOp {
        let kind = match &node.kind {
            ast::DimOpKind::Add => hir::DimOpKind::Add,
            ast::DimOpKind::Sub => hir::DimOpKind::Sub,
            ast::DimOpKind::Mul => hir::DimOpKind::Mul,
            ast::DimOpKind::Div => hir::DimOpKind::Div,
        };
        hir::DimOp { kind }
    },
}

/// Lower the fields of a struct.
/// NB: Records an error if two fields overlap.
impl<'i, A, B> Lower<VecMap<hir::Name, B>, Context<'i>> for Vec<ast::Field<A>>
where
    A: Lower<B, Context<'i>> + std::fmt::Debug,
{
    fn lower(&self, ctx: &mut Context<'i>) -> VecMap<hir::Name, B> {
        use arc_script_core_shared::VecMapEntry;
        let mut map: VecMap<hir::Name, B> = VecMap::new();
        for f in self {
            if let VecMapEntry::Vacant(entry) = map.entry(f.name) {
                entry.insert(f.val.lower(ctx));
            } else {
                ctx.diags.intern(Error::FieldClash { name: f.name });
            }
        }
        map
    }
}
