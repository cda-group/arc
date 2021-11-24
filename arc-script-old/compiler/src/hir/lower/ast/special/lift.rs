use crate::hir;
use crate::hir::FunKind::Free;
use crate::hir::Name;
use arc_script_compiler_shared::Set;
use arc_script_compiler_shared::VecMap;

use super::Context;

/// Attempts to lifts a block into a function. While lambda lifting replaces
/// the block with a pointer to that function, this type of lifting replaces
/// the block with a call to that function. Lifting is only possible if the
/// block does not contain any control-flow constructs. If lifting fails, the
/// original block is returned.
pub(crate) fn lift(block: hir::Block, ctx: &mut Context<'_>) -> hir::Block {
    let mut vars = Set::default();
    match block.fv(&mut vars, ctx) {
        Liftable::No => block,
        Liftable::Yes => {
            let path = ctx.fresh_path();
            let vars = vars.into_iter().collect::<Vec<_>>();

            let (params, args): (Vec<_>, Vec<_>) = vars
                .iter()
                .map(|x| {
                    let t = ctx.types.fresh();
                    let p = ctx.new_param(*x, t);
                    let v = ctx.new_var(*x, hir::ScopeKind::Local);
                    (p, v)
                })
                .unzip();

            let its = args.iter().map(|e| e.t).collect();
            let rt = block.var.t;
            let t = ctx.types.intern(hir::TypeKind::Fun(its, rt));
            let loc = block.loc;

            let item = hir::Item::new(
                hir::ItemKind::Fun(hir::Fun::new(path, Free, params, block, t, rt)),
                loc,
            );

            ctx.hir.intern(path, item);
            ctx.hir.namespace.push(path.id);

            let kind0 = hir::ExprKind::Item(path);
            let id0 = ctx.hir.exprs.intern(kind0);
            let e0 = hir::Expr::syn(id0, t);
            let (s0, v0) = ctx.new_stmt_assign_var(e0);

            let kind1 = hir::ExprKind::Call(v0, args);
            let id1 = ctx.hir.exprs.intern(kind1);
            let e1 = hir::Expr::syn(id1, rt);
            let (s1, v1) = ctx.new_stmt_assign_var(e1);

            hir::Block::syn(vec![s0, s1].into(), v1)
        }
    }
}

/// Represents whether a block is liftable or not.
enum Liftable {
    Yes,
    No,
}

impl std::ops::FromResidual for Liftable {
    fn from_residual(residual: <Self as std::ops::Try>::Residual) -> Self {
        Self::No
    }
}

impl std::ops::Try for Liftable {
    type Output = ();

    type Residual = ();

    fn from_output(output: Self::Output) -> Self {
        Self::Yes
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Liftable::Yes => std::ops::ControlFlow::Continue(()),
            Liftable::No => std::ops::ControlFlow::Break(()),
        }
    }
}

trait FreeVars {
    /// Mutably constructs a set of free-variables. Returns `Liftable::Yes` if the expression can
    /// be lifted, else `Liftable::No` (if control-flow constructs were found encountered).
    fn fv(&self, set: &mut Set<Name>, ctx: &Context<'_>) -> Liftable;
}

macro_rules! freevars {
    {
        [$node:ident, $set:ident, $ctx:ident]

        $($ty:ty => $expr:expr,)*
    } => {
        $(
            impl FreeVars for $ty {
                fn fv(&self, $set: &mut Set<Name>, $ctx: &Context<'_>) -> Liftable {
                    let $node = self;
                    $expr;
                    Liftable::Yes
                }
            }
        )*
    }
}

freevars! {
    [node, set, ctx]

    // NOTE: Here we go over all variables backwards
    hir::Block => {
        node.var.fv(set, ctx)?;
        for stmt in node.stmts.iter().rev() {
            stmt.fv(set, ctx)?;
        }
    },
    hir::Stmt => match &node.kind {
        hir::StmtKind::Assign(item) => item.fv(set, ctx),
    },
    hir::Assign => {
        node.expr.fv(set, ctx)?;
        if let hir::ParamKind::Ok(x) = &node.param.kind {
            set.remove(x);
        }
    },
    hir::Var => if let hir::VarKind::Ok(x, _) = &node.kind {
        set.insert(*x);
    },
    hir::Expr => match ctx.hir.exprs.resolve(node) {
        hir::ExprKind::Return(_) => Liftable::No?,
        hir::ExprKind::Break(_) => Liftable::No?,
        hir::ExprKind::Continue => Liftable::No?,
        hir::ExprKind::Lit(_) => {}
        hir::ExprKind::Array(vs) => vs.fv(set, ctx)?,
        hir::ExprKind::Struct(vfs) => vfs.fv(set, ctx)?,
        hir::ExprKind::Enwrap(_, v) => v.fv(set, ctx)?,
        hir::ExprKind::Unwrap(_, v) => v.fv(set, ctx)?,
        hir::ExprKind::Is(_, v) => v.fv(set, ctx)?,
        hir::ExprKind::Tuple(vs) => vs.fv(set, ctx)?,
        hir::ExprKind::Item(_) => {}
        hir::ExprKind::UnOp(_, v) => {
            v.fv(set, ctx)?;
        }
        hir::ExprKind::BinOp(v0, _, v1) => {
            v0.fv(set, ctx)?;
            v1.fv(set, ctx)?;
        }
        hir::ExprKind::If(v, b0, b1) => {
            v.fv(set, ctx)?;
            b0.fv(set, ctx)?;
            b1.fv(set, ctx)?;
        }
        hir::ExprKind::Call(v, vs) => {
            v.fv(set, ctx)?;
            vs.fv(set, ctx)?;
        }
        hir::ExprKind::SelfCall(x, vs) => vs.fv(set, ctx)?,
        hir::ExprKind::Invoke(v, _, vs) => {
            v.fv(set, ctx)?;
            vs.fv(set, ctx)?;
        }
        hir::ExprKind::Select(v, vs) => {
            v.fv(set, ctx)?;
            vs.fv(set, ctx)?;
        }
        hir::ExprKind::Emit(v) => {
            v.fv(set, ctx)?;
        }
        hir::ExprKind::Loop(v) => {
            v.fv(set, ctx)?;
        }
        hir::ExprKind::Access(v, _) => v.fv(set, ctx)?,
        hir::ExprKind::Log(v) => v.fv(set, ctx)?,
        hir::ExprKind::Project(v, _) => v.fv(set, ctx)?,
        hir::ExprKind::After(v, b) => {
            v.fv(set, ctx)?;
            b.fv(set, ctx)?;
        }
        hir::ExprKind::Every(v, b) => {
            v.fv(set, ctx)?;
            b.fv(set, ctx)?;
        }
        hir::ExprKind::Cast(v, _) => v.fv(set, ctx)?,
        hir::ExprKind::Unreachable => {}
        hir::ExprKind::Initialise(_, v) => v.fv(set, ctx)?,
        hir::ExprKind::Err => {}
    },
}

impl<T: FreeVars> FreeVars for Vec<T> {
    fn fv(&self, set: &mut Set<Name>, ctx: &Context<'_>) -> Liftable {
        self.iter().try_for_each(|x| x.fv(set, ctx))
    }
}

impl<T: FreeVars> FreeVars for VecMap<Name, T> {
    fn fv(&self, set: &mut Set<Name>, ctx: &Context<'_>) -> Liftable {
        self.values().try_for_each(|x| x.fv(set, ctx))
    }
}
