use crate::ast::*;
use crate::info::Info;
use crate::symbols::SymbolTable;
use ExprKind::*;
use UnOpKind::*;

impl SyntaxTree {
    pub fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr(f));
        self.body.for_each_expr(f);
    }

    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|(_, fundef)| f(fundef));
    }

    pub fn for_each_decl<F: FnMut(&mut Decl)>(&mut self, ref mut f: F, table: &mut SymbolTable) {
        table.decls.iter_mut().for_each(f);
    }
}

impl Expr {
    fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, f: &mut F) {
        f(self);
        match &mut self.kind {
            If(c, t, e) => {
                c.for_each_expr(f);
                t.for_each_expr(f);
                e.for_each_expr(f);
            }
            Match(e, cases) => {
                e.for_each_expr(f);
                cases.iter_mut().for_each(|(_, e)| e.for_each_expr(f));
            }
            Let(_, v) => v.for_each_expr(f),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.for_each_expr(f)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr(f);
                r.for_each_expr(f);
            }
            UnOp(op, e) => {
                op.for_each_expr(f);
                e.for_each_expr(f);
            }
            Sink(_) => {}
            Source(_) => {}
            Loop(cond, body) => {
                cond.for_each_expr(f);
                body.for_each_expr(f);
            }
            Closure(_, body) => body.for_each_expr(f),
            ExprErr => {}
        }
    }
}

impl UnOpKind {
    fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, f: &mut F) {
        match self {
            FunCall(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            MethodCall(_, ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            _ => {}
        }
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
