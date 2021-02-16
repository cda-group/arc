use crate::compiler::hir;
use crate::compiler::hir::Name;
use arc_script_core_shared::Map;
use arc_script_core_shared::VecMap;

use super::Context;

/// Attempts to lifts an expression into a function. While lambda lifting replaces
/// the expression with a pointer to that function, this type of lifting replaces
/// the expression with a call to that function. Lifting is only possible if the
/// expression does not contain any control-flow constructs. If lifting fails, the
/// original expression is returned.
pub(super) fn lift(body: hir::Expr, ctx: &mut Context<'_>) -> hir::Expr {
    let mut vars = Map::default();
    if body.fv(&mut vars).is_err() {
        body
    } else {
        let (_, path) = ctx.info.fresh_name_path();
        let vars = vars.into_iter().collect::<Vec<_>>();

        let params = vars
            .iter()
            .map(|(x, e)| hir::Param::syn(hir::ParamKind::Var(*x), e.tv))
            .collect::<Vec<_>>();

        let args = vars.into_iter().map(|(_, e)| e).collect::<Vec<_>>();

        let itvs = args.iter().map(|arg| arg.tv).collect();
        let otv = body.tv;
        let tv = ctx.info.types.intern(hir::TypeKind::Fun(itvs, otv));
        let loc = body.loc;

        let item = hir::Item::new(
            hir::ItemKind::Fun(hir::Fun::new(path, params, None, body, tv, otv)),
            loc,
        );

        ctx.hir.defs.insert(path, item);
        ctx.hir.items.push(path);

        hir::Expr::syn(
            hir::ExprKind::Call(hir::Expr::syn(hir::ExprKind::Item(path), tv).into(), args),
            otv,
        )
    }
}

trait FreeVars {
    /// Mutably constructs a set of free-variables. Returns `Ok(())` if no
    /// control-flow were encountered, else `Err(())`
    fn fv(&self, union: &mut Map<Name, hir::Expr>) -> Result<(), ()>;
}

impl FreeVars for hir::Expr {
    fn fv(&self, union: &mut Map<Name, hir::Expr>) -> Result<(), ()> {
        match &self.kind {
            hir::ExprKind::Let(x, e0, e1) => {
                e0.fv(union)?;
                e1.fv(union)?;
                if let hir::ParamKind::Var(x) = &x.kind {
                    union.remove(x);
                }
            }
            hir::ExprKind::Var(x) => {
                union.insert(*x, self.clone());
            }
            hir::ExprKind::Lit(_) => {}
            hir::ExprKind::Array(es) => es.fv(union)?,
            hir::ExprKind::Struct(efs) => efs.fv(union)?,
            hir::ExprKind::Enwrap(_, e) => e.fv(union)?,
            hir::ExprKind::Unwrap(_, e) => e.fv(union)?,
            hir::ExprKind::Is(_, e) => e.fv(union)?,
            hir::ExprKind::Tuple(es) => es.fv(union)?,
            hir::ExprKind::Item(_) => {}
            hir::ExprKind::UnOp(_, e) => {
                e.fv(union)?;
            }
            hir::ExprKind::BinOp(e0, _, e1) => {
                e0.fv(union)?;
                e1.fv(union)?;
            }
            hir::ExprKind::If(e0, e1, e2) => {
                e0.fv(union)?;
                e1.fv(union)?;
                e2.fv(union)?;
            }
            hir::ExprKind::Call(e, es) => {
                e.fv(union)?;
                es.fv(union)?;
            }
            hir::ExprKind::Emit(e) => {
                e.fv(union)?;
            }
            hir::ExprKind::Loop(e) => {
                e.fv(union)?;
            }
            hir::ExprKind::Access(e, _) => e.fv(union)?,
            hir::ExprKind::Log(e) => e.fv(union)?,
            hir::ExprKind::Project(e, _) => e.fv(union)?,
            /// NOTE: Control-flow constructs make lifting impossible.
            hir::ExprKind::Return(_) => return Err(()),
            hir::ExprKind::Break => return Err(()),
            hir::ExprKind::Todo => {}
            hir::ExprKind::Err => {}
        }
        Ok(())
    }
}

impl<T: FreeVars> FreeVars for Vec<T> {
    fn fv(&self, union: &mut Map<Name, hir::Expr>) -> Result<(), ()> {
        self.iter().try_for_each(|x| x.fv(union))
    }
}

impl<T: FreeVars> FreeVars for VecMap<Name, T> {
    fn fv(&self, union: &mut Map<Name, hir::Expr>) -> Result<(), ()> {
        self.values().try_for_each(|x| x.fv(union))
    }
}
