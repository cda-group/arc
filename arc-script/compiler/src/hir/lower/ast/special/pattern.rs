use super::Context;
use crate::ast;
use crate::hir;
use crate::hir::HIR;
use crate::info::diags::Error;
use crate::info::files::Loc;
use crate::info::types::TypeId;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;

use crate::hir::lower::ast::special::path;

impl Context<'_> {
    fn create_binding(&mut self, e: hir::Expr, cases: &mut VecDeque<Case>) -> hir::Var {
        let (s, v) = e.into_stmt_var(self);
        cases.push_back(Case::Stmt(s));
        v
    }
    fn create_guard(&mut self, e: hir::Var, cases: &mut VecDeque<Case>) {
        cases.push_back(Case::Guard(e));
    }
}

/// Lowers a list of parameters which may contain patterns into a list of primitive parameters which
/// contain no patterns. A list of cases are also returned for unfolding the patterns.
pub(crate) fn lower_params(
    params: &[ast::Param],
    kind: hir::ScopeKind,
    ctx: &mut Context<'_>,
) -> (Vec<hir::Param>, VecDeque<Case>) {
    let mut cases = VecDeque::new();

    let params = params
        .iter()
        .map(|param| {
            param.pat.lower(
                ctx.new_type_fresh_if_none(&param.t),
                kind,
                false,
                &mut cases,
                ctx,
            )
        })
        .collect();

    (params, cases)
}

pub(crate) fn lower_assign(
    assign: &ast::Assign,
    is_refutable: bool,
    kind: hir::ScopeKind,
    ctx: &mut Context<'_>,
) -> VecDeque<Case> {
    let e = assign.expr.lower(ctx);
    let mut cases = VecDeque::new();

    // TODO: What to do with type?
    assign.param.pat.flatten(e, is_refutable, ctx, &mut cases);

    cases
}

/// Lowers a single `<pat>`, e.g., `on <pat> => <expr>`
pub(crate) fn lower_pat(
    pat: &ast::Pat,
    is_refutable: bool,
    kind: hir::ScopeKind,
    ctx: &mut Context<'_>,
) -> (hir::Param, VecDeque<Case>) {
    let mut cases = VecDeque::new();

    let param = pat.lower(ctx.types.fresh(), kind, is_refutable, &mut cases, ctx);

    (param, cases)
}

impl Context<'_> {
    /// Folds cases into a block. The `else_block` should be specified if the
    /// pattern is refutable (i.e., contains guards).
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn fold_cases(
        &mut self,
        then_block: hir::Block,
        else_block: hir::Block,
        cases: VecDeque<Case>,
    ) -> hir::Block {
        cases
            .into_iter()
            .rev()
            .fold(then_block, |mut then_block, case| match case {
                Case::Guard(v) => self
                    .new_expr_if(v, then_block, else_block.clone())
                    .into_block(self),
                Case::Stmt(s) => {
                    then_block.stmts.push_front(s);
                    then_block
                }
            })
    }
}

pub(crate) fn cases_to_stmts(cases: VecDeque<Case>) -> VecDeque<hir::Stmt> {
    cases
        .into_iter()
        .map(|case| match case {
            Case::Guard(e) => unreachable!(),
            Case::Stmt(s) => s,
        })
        .collect()
}

#[derive(Debug)]
pub(crate) enum Case {
    Guard(hir::Var),
    Stmt(hir::Stmt),
}

impl ast::Pat {
    /// Lowers a pattern into a single parameter and cases for unfolding the parameter.
    fn lower(
        &self,
        t: hir::Type,
        kind: hir::ScopeKind,
        is_refutable: bool,
        cases: &mut VecDeque<Case>,
        ctx: &mut Context<'_>,
    ) -> hir::Param {
        let kind = if let ast::PatKind::Var(x) = ctx.ast.pats.resolve(self) {
            ctx.res
                .stack
                .rename_to_unique(*x, kind, ctx.info)
                .map_or(hir::ParamKind::Err, hir::ParamKind::Ok)
        } else {
            // The top pattern is not a variable and needs to be flattened.
            // Start by introducing a new variable for the pattern.
            let x = ctx.names.fresh();
            let v = ctx.new_var(x, kind);
            self.flatten(v, is_refutable, ctx, cases);
            hir::ParamKind::Ok(x)
        };
        hir::Param::syn(kind, t)
    }

    /// Unfolds a pattern into a list of cases.
    fn flatten(
        &self,
        // The ssa-variable the pattern is matching on
        v_rhs: hir::Var,
        is_refutable: bool,
        ctx: &mut Context<'_>,
        cases: &mut VecDeque<Case>,
    ) {
        match ctx.ast.pats.resolve(self) {
            // Irrefutable patterns
            ast::PatKind::Struct(fs) => {
                for f in fs {
                    let (s, v) = ctx.new_expr_access(v_rhs, f.name).into_stmt_var(ctx);
                    cases.push_back(Case::Stmt(s));
                    if let Some(p) = &f.val {
                        // If the struct has a field-pattern
                        p.flatten(v, is_refutable, ctx, cases);
                    } else {
                        let (x, _) = get!(&v.kind, hir::VarKind::Ok(x, k));
                        ctx.res.stack.rename(f.name, || *x, hir::ScopeKind::Local);
                    }
                }
            }
            // p0 by p1 ==> {val:p0, key:p1} ==> let x0 = e0; let p1 = x0.val; let p1 = x0.key; ...
            ast::PatKind::By(p0, p1) => {
                let (s0, v0) = ctx
                    .new_expr_access(v_rhs, ctx.names.common.val)
                    .into_stmt_var(ctx);
                let (s1, v1) = ctx
                    .new_expr_access(v_rhs, ctx.names.common.key)
                    .into_stmt_var(ctx);

                cases.push_back(Case::Stmt(s0));
                cases.push_back(Case::Stmt(s1));

                p0.flatten(v0, is_refutable, ctx, cases);
                p1.flatten(v1, is_refutable, ctx, cases);
            }
            ast::PatKind::Var(x0) => {
                if let hir::VarKind::Ok(x1, _) = &v_rhs.kind {
                    if ctx
                        .res
                        .stack
                        .rename(*x0, || *x1, hir::ScopeKind::Local)
                        .is_none()
                    {
                        ctx.diags.intern(Error::NameClash { name: *x0 });
                    }
                }
            }
            ast::PatKind::Tuple(ps) => {
                for (i, p) in ps.iter().enumerate() {
                    let (s0, v0) = ctx.new_expr_project(v_rhs, i).into_stmt_var(ctx);
                    cases.push_back(Case::Stmt(s0));
                    p.flatten(v0, is_refutable, ctx, cases)
                }
            }
            ast::PatKind::Ignore => {}
            ast::PatKind::Err => {}
            ast::PatKind::Or(_p0, _p1) => todo!(),
            ast::PatKind::Const(kind) if is_refutable => {
                let (s0, v0) = ctx.new_expr_lit(kind).into_stmt_var(ctx);
                cases.push_back(Case::Stmt(s0));
                let (s1, v1) = ctx.new_expr_equ(v_rhs, v0).into_stmt_var(ctx);
                cases.push_back(Case::Stmt(s1));
                cases.push_back(Case::Guard(v1));
            }
            ast::PatKind::Variant(x, p) if is_refutable => {
                if let Some(x) = path::lower_variant_path(x, ctx) {
                    let (s0, v0) = ctx.new_expr_is(x, v_rhs).into_stmt_var(ctx);
                    cases.push_back(Case::Stmt(s0));
                    cases.push_back(Case::Guard(v0));
                    let (s1, v1) = ctx.new_expr_unwrap(x, v_rhs).into_stmt_var(ctx);
                    cases.push_back(Case::Stmt(s1));
                    p.flatten(v1, is_refutable, ctx, cases);
                }
            }
            _ => ctx.diags.intern(Error::RefutablePattern { loc: self.loc }),
        }
    }

    /// Returns the parameters of a pattern
    fn params(&self, params: &mut Vec<hir::Param>, ctx: &mut Context<'_>) {
        match ctx.ast.pats.resolve(self) {
            ast::PatKind::Ignore => {}
            ast::PatKind::Struct(pfs) => pfs.iter().for_each(|f| {
                if let Some(p) = &f.val {
                    p.params(params, ctx)
                } else {
                    let t = ctx.types.fresh();
                    let param = hir::Param::new(hir::ParamKind::Ok(f.name), t, self.loc);
                    params.push(param);
                }
            }),
            ast::PatKind::Tuple(ps) => ps.iter().for_each(|p| p.params(params, ctx)),
            ast::PatKind::Var(x) => {
                let t = ctx.types.fresh();
                let param = hir::Param::new(hir::ParamKind::Ok(*x), t, self.loc);
                params.push(param);
            }
            ast::PatKind::Const(_) => unreachable!(),
            ast::PatKind::Or(_, _) => unreachable!(),
            ast::PatKind::By(p0, p1) => {
                p0.params(params, ctx);
                p1.params(params, ctx);
            }
            ast::PatKind::Variant(_, _) => {}
            ast::PatKind::Err => {}
        }
    }
}

impl Context<'_> {
    fn extract_params(&mut self, ps: &[ast::Param]) -> Vec<hir::Param> {
        let mut params = Vec::new();
        ps.iter().for_each(|p| p.pat.params(&mut params, self));
        params
    }
}

pub(crate) fn params_to_args(
    params: &[hir::Param],
    kind: hir::ScopeKind,
    ctx: &mut Context<'_>,
) -> Vec<hir::Var> {
    params
        .iter()
        .map(|p| {
            let x = get!(p.kind, hir::ParamKind::Ok(x));
            ctx.new_var(x, hir::ScopeKind::Local)
        })
        .collect::<Vec<_>>()
}
