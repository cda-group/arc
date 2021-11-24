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

impl hir::Expr {
    pub(crate) fn into_stmt_var(self, ctx: &mut Context<'_>) -> (hir::Stmt, hir::Var) {
        ctx.new_stmt_assign_var(self)
    }

    pub(crate) fn into_stmt(self, p: hir::Param, ctx: &mut Context<'_>) -> hir::Stmt {
        let a = hir::Assign::new(hir::MutKind::Immutable, p, self);
        ctx.new_stmt(hir::StmtKind::Assign(a))
    }

    /// Extends the current block with an assignment to the expression and returns a variable
    /// referencing the assignment. For example, `1` extends the block with `val x0 = 1;` and
    /// returns `x0`.
    pub(crate) fn into_ssa(self, ctx: &mut Context<'_>) -> hir::Var {
        let (s, v) = self.into_stmt_var(ctx);
        ctx.get_stmts().push_back(s);
        v
    }

    pub(crate) fn into_block(self, ctx: &mut Context<'_>) -> hir::Block {
        let (s, v) = ctx.new_stmt_assign_var(self);
        hir::Block::syn(vec![s].into(), v)
    }
}

impl Context<'_> {
    pub(crate) fn new_var(&mut self, x: hir::Name, scope: hir::ScopeKind) -> hir::Var {
        let t = self.types.fresh();
        hir::Var::syn(hir::VarKind::Ok(x, scope), t)
    }

    pub(crate) fn new_param(&mut self, x: hir::Name, t: hir::Type) -> hir::Param {
        hir::Param::new(hir::ParamKind::Ok(x), t, Loc::Fake)
    }

    pub(crate) fn new_var_err(&mut self, loc: Loc) -> hir::Var {
        hir::Var::new(hir::VarKind::Err, self.types.fresh(), loc)
    }

    pub(crate) fn new_fresh_param_var(&mut self) -> (hir::Param, hir::Var) {
        let x = self.names.fresh();
        let t = self.types.fresh();
        let v = self.new_var(x, hir::ScopeKind::Local);
        let p = self.new_param(x, t);
        (p, v)
    }

    pub(crate) fn new_expr_access(&mut self, v: hir::Var, x: hir::Name) -> hir::Expr {
        self.new_expr(hir::ExprKind::Access(v, x))
    }

    pub(crate) fn new_expr_project(&mut self, v: hir::Var, i: impl Into<hir::Index>) -> hir::Expr {
        self.new_expr(hir::ExprKind::Project(v, i.into()))
    }

    pub(crate) fn new_expr_lit(&mut self, kind: &ast::LitKind) -> hir::Expr {
        let kind = kind.lower(self);
        self.new_expr(hir::ExprKind::Lit(kind))
    }

    pub(crate) fn new_expr_equ(&mut self, v0: hir::Var, v1: hir::Var) -> hir::Expr {
        let op = hir::BinOp::syn(hir::BinOpKind::Equ);
        self.new_expr(hir::ExprKind::BinOp(v0, op, v1))
    }

    pub(crate) fn new_expr_is(&mut self, x: hir::Path, v: hir::Var) -> hir::Expr {
        self.new_expr(hir::ExprKind::Is(x, v))
    }

    pub(crate) fn new_expr_unwrap(&mut self, x: hir::Path, v: hir::Var) -> hir::Expr {
        self.new_expr(hir::ExprKind::Unwrap(x, v))
    }

    pub(crate) fn new_expr_enwrap(&mut self, x: hir::Path, v: hir::Var) -> hir::Expr {
        self.new_expr(hir::ExprKind::Enwrap(x, v))
    }

    pub(crate) fn new_expr_if(&mut self, v: hir::Var, b0: hir::Block, b1: hir::Block) -> hir::Expr {
        self.new_expr(hir::ExprKind::If(v, b0, b1))
    }

    pub(crate) fn new_stmt(&mut self, kind: hir::StmtKind) -> hir::Stmt {
        hir::Stmt::syn(kind)
    }

    pub(crate) fn new_stmt_assign(&mut self, p: hir::Param, e: hir::Expr) -> hir::Stmt {
        let a = hir::Assign::new(hir::MutKind::Immutable, p, e);
        self.new_stmt(hir::StmtKind::Assign(a))
    }

    pub(crate) fn new_stmt_assign_ignore(&mut self, e: hir::Expr) -> hir::Stmt {
        let t = self.types.fresh();
        let p = hir::Param::syn(hir::ParamKind::Ignore, t);
        self.new_stmt_assign(p, e)
    }

    pub(crate) fn new_stmt_assign_var(&mut self, e: hir::Expr) -> (hir::Stmt, hir::Var) {
        let x = self.info.names.fresh();
        let t = self.types.fresh();
        let p = hir::Param::syn(hir::ParamKind::Ok(x), t);
        let a = hir::Assign::new(hir::MutKind::Immutable, p, e);
        let s = hir::Stmt::syn(hir::StmtKind::Assign(a));
        let v = self.new_var(x, hir::ScopeKind::Local);
        (s, v)
    }

    pub(crate) fn new_expr_unit(&mut self) -> hir::Expr {
        self.new_expr(hir::ExprKind::Lit(hir::LitKind::Unit))
    }

    pub(crate) fn new_expr_unreachable(&mut self) -> hir::Expr {
        self.new_expr(hir::ExprKind::Unreachable)
    }

    pub(crate) fn new_expr(&mut self, kind: hir::ExprKind) -> hir::Expr {
        let id = self.hir.exprs.intern(kind);
        let t = self.types.fresh();
        hir::Expr::syn(id, t)
    }

    pub(crate) fn new_expr_with_loc(&mut self, kind: hir::ExprKind, loc: Loc) -> hir::Expr {
        let id = self.hir.exprs.intern(kind);
        let t = self.types.fresh();
        hir::Expr::new(id, t, loc)
    }

    pub(crate) fn new_type_unit_if_none(&mut self, t: &Option<ast::Type>) -> hir::Type {
        if let Some(t) = t {
            t.lower(self)
        } else {
            self.types.intern(hir::ScalarKind::Unit)
        }
    }

    pub(crate) fn new_type_unit(&mut self) -> hir::Type {
        self.types.intern(hir::ScalarKind::Unit)
    }

    pub(crate) fn new_type_fresh_if_none(&mut self, t: &Option<ast::Type>) -> hir::Type {
        if let Some(t) = t {
            t.lower(self)
        } else {
            self.types.fresh()
        }
    }

    /// Creates an implicit block around an expression
    pub(crate) fn new_implicit_block(&mut self, e: &ast::Expr) -> hir::Block {
        self.res.stack.push_scope();
        let v = e.lower(self);
        let stmts = self.res.stack.pop_scope();
        hir::Block::syn(stmts, v)
    }

    /// Creates a unit block if the provided block is `None`
    pub(crate) fn new_block_unit_if_none(&mut self, b: &Option<ast::Block>) -> hir::Block {
        if let Some(b) = b {
            b.lower(self)
        } else {
            let e = self.new_expr_unit();
            let (s, e) = self.new_stmt_assign_var(e);
            hir::Block::syn(vec![s].into(), e)
        }
    }

    /// Extends the current path with a name and returns the new path.
    pub(crate) fn new_path(&mut self, name: hir::Name) -> hir::Path {
        let path = self.res.path;
        self.paths.intern_child(path, name).into()
    }

    /// Returns a dummy-path, i.e., `<path>::__`.
    pub(crate) fn new_path_dummy(&mut self, path: hir::Path) -> hir::Path {
        let dummy = self.names.common.dummy;
        self.paths.intern_child(path, dummy).into()
    }

    pub(crate) fn new_fun(
        &mut self,
        path: hir::Path,
        params: Vec<hir::Param>,
        body: hir::Block,
    ) -> hir::Fun {
        hir::Fun {
            kind: hir::FunKind::Free,
            path,
            params,
            body,
            t: self.types.fresh(),
            rt: self.types.fresh(),
        }
    }

    pub(crate) fn new_method(
        &mut self,
        path: hir::Path,
        params: Vec<hir::Param>,
        body: hir::Block,
    ) -> hir::Fun {
        hir::Fun {
            kind: hir::FunKind::Method,
            path,
            params,
            body,
            t: self.types.fresh(),
            rt: self.types.fresh(),
        }
    }

    /// Returns the a mutable reference to the statements of the innermost scope.
    pub(crate) fn get_stmts(&mut self) -> &mut VecDeque<hir::Stmt> {
        self.res.stack.stmts()
    }
}

impl hir::Block {
    /// Prepends statements to a block.
    pub(crate) fn prepend_stmts(&mut self, mut stmts: VecDeque<hir::Stmt>) {
        stmts.append(&mut self.stmts);
        self.stmts = stmts;
    }
}
