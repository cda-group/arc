use crate::info::Info;
use crate::prelude::*;
use crate::symbols::SymbolTable;

impl SyntaxTree {
    // Post-order traversal (Leaves first)
    pub fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr(f));
        self.body.for_each_expr(f);
    }
    // Pre-order traversal (Parents first)
    pub fn for_each_expr_preorder<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr_preorder(f));
        self.body.for_each_expr_preorder(f)
    }

    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|(_, fundef)| f(fundef));
    }

    pub fn for_each_decl<F: FnMut(&mut Decl)>(&mut self, ref mut f: F, table: &mut SymbolTable) {
        table.decls.iter_mut().for_each(f);
    }

    pub fn for_each_task<F: FnMut(&mut TaskDef)>(&mut self, ref mut f: F) {
        self.taskdefs.iter_mut().for_each(|(_, taskdef)| f(taskdef));
    }
}

impl Expr {
    fn for_each_expr_preorder<F: FnMut(&mut Expr)>(&mut self, f: &mut F) {
        f(self);
        match &mut self.kind {
            If(c, t, e) => {
                c.for_each_expr_preorder(f);
                t.for_each_expr_preorder(f);
                e.for_each_expr_preorder(f);
            }
            For(_, e, c, b) => {
                e.for_each_expr_preorder(f);
                if let Some(c) = c {
                    c.for_each_expr_preorder(f);
                }
                b.for_each_expr_preorder(f);
            }
            Match(e, cases) => {
                e.for_each_expr_preorder(f);
                cases
                    .iter_mut()
                    .for_each(|(_, e)| e.for_each_expr_preorder(f));
            }
            Let(_, v) => v.for_each_expr_preorder(f),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_preorder(f)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_preorder(f)),
            ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.for_each_expr_preorder(f)),
            ConsEnum(vs) => vs.iter_mut().for_each(|(_, v)| v.for_each_expr_preorder(f)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr_preorder(f);
                r.for_each_expr_preorder(f);
            }
            UnOp(Call(ps), e) => {
                ps.iter_mut().for_each(|p| p.for_each_expr_preorder(f));
                e.for_each_expr_preorder(f);
            }
            UnOp(_, _) => {}
            Sink(_) => {}
            Source(_) => {}
            Loop(cond, body) => {
                cond.for_each_expr_preorder(f);
                body.for_each_expr_preorder(f);
            }
            Closure(_, body) => body.for_each_expr_preorder(f),
            ExprErr => {}
        }
    }
}

impl TaskDef {
    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|(_, fundef)| f(fundef));
    }
}

impl Expr {
    fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, f: &mut F) {
        match &mut self.kind {
            If(c, t, e) => {
                c.for_each_expr(f);
                t.for_each_expr(f);
                e.for_each_expr(f);
            }
            For(_, e, c, b) => {
                e.for_each_expr(f);
                if let Some(c) = c {
                    c.for_each_expr(f);
                }
                b.for_each_expr(f);
            }
            Match(e, cases) => {
                e.for_each_expr(f);
                cases.iter_mut().for_each(|(_, e)| e.for_each_expr(f));
            }
            Let(_, v) => v.for_each_expr(f),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.for_each_expr(f)),
            ConsEnum(vs) => vs.iter_mut().for_each(|(_, v)| v.for_each_expr(f)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr(f);
                r.for_each_expr(f);
            }
            UnOp(Call(ps), e) => {
                ps.iter_mut().for_each(|p| p.for_each_expr(f));
                e.for_each_expr(f);
            }
            UnOp(_, _) => {}
            Sink(_) => {}
            Source(_) => {}
            Loop(cond, body) => {
                cond.for_each_expr(f);
                body.for_each_expr(f);
            }
            Closure(_, body) => body.for_each_expr(f),
            ExprErr => {}
        }
        f(self);
    }
}

pub struct Printer<'a> {
    pub info: &'a Info<'a>,
    pub tabs: u32,
    pub verbose: bool,
}

const TAB: &str = "    ";

impl Printer<'_> {
    pub fn indent(&self) -> String {
        format!("\n{}", (0..self.tabs).map(|_| TAB).collect::<String>())
    }

    pub fn tab(&self) -> Printer {
        Printer {
            info: self.info,
            tabs: self.tabs + 1,
            verbose: self.verbose,
        }
    }

    pub fn untab(&self) -> Printer {
        Printer {
            info: self.info,
            tabs: self.tabs - 1,
            verbose: self.verbose,
        }
    }
}

pub fn merge<T>(mut a: Vec<T>, mut b: Vec<T>) -> Vec<T> {
    a.append(&mut b);
    a
}
