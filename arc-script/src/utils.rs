use crate::ast::*;
use crate::info::Info;
use crate::symbols::SymbolTable;
use DimKind::*;
use ExprKind::*;
use ShapeKind::*;
use TypeKind::*;
use UnOpKind::*;

impl Expr {
    pub fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, mut fun: F) {
        self.for_each_expr_rec(&mut fun);
    }

    pub fn for_each_type<F: FnMut(&mut Type)>(&mut self, mut f: F, table: &mut SymbolTable) {
        self.for_each_expr(|e| {
            e.ty.for_each_type_rec(&mut f);
            match &mut e.kind {
                Let(id, _, _) => table.get_mut(id).ty.for_each_type_rec(&mut f),
                UnOp(Cast(ty), _) => ty.for_each_type_rec(&mut f),
                _ => {}
            }
        })
    }

    pub fn for_each_dim_expr<F: FnMut(&mut Expr)>(&mut self, mut f: F, table: &mut SymbolTable) {
        self.for_each_type(
            |ty| {
                if let Array(_, shape) = &mut ty.kind {
                    match &mut shape.kind {
                        Ranked(dims) => dims.iter_mut().for_each(|dim| match &mut dim.kind {
                            Symbolic(expr) => expr.for_each_expr_rec(&mut f),
                            Hole => {}
                        }),
                        Unranked => {}
                    }
                }
            },
            table,
        )
    }

    fn for_each_expr_rec<F: FnMut(&mut Expr)>(&mut self, fun: &mut F) {
        fun(self);
        match &mut self.kind {
            If(c, t, e) => {
                c.for_each_expr_rec(fun);
                t.for_each_expr_rec(fun);
                e.for_each_expr_rec(fun);
            }
            Match(e, cases) => {
                e.for_each_expr_rec(fun);
                cases.iter_mut().for_each(|(_, e)| e.for_each_expr_rec(fun));
            }
            Let(_, v, b) => {
                v.for_each_expr_rec(fun);
                b.for_each_expr_rec(fun);
            }
            FunCall(_, ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun)),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun)),
            ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.for_each_expr_rec(fun)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr_rec(fun);
                r.for_each_expr_rec(fun);
            }
            UnOp(_, e) => e.for_each_expr_rec(fun),
            ExprErr => {}
        }
    }
}
impl Type {
    pub fn for_each_type<F: FnMut(&mut Type)>(&mut self, mut f: F) {
        self.for_each_type_rec(&mut f);
    }

    fn for_each_type_rec<F: FnMut(&mut Type)>(&mut self, f: &mut F) {
        f(self);
        match &mut self.kind {
            Array(ty, _) => ty.for_each_type_rec(f),
            Struct(fs) => fs.iter_mut().for_each(|(_, ty)| ty.for_each_type_rec(f)),
            Tuple(tys) => tys.iter_mut().for_each(|ty| ty.for_each_type_rec(f)),
            Optional(ty) => ty.for_each_type_rec(f),
            Scalar(_) => {}
            Unknown => {}
            Fun(tys, ty) => {
                tys.iter_mut().for_each(|ty| ty.for_each_type_rec(f));
                ty.for_each_type_rec(f);
            }
            TypeErr => {}
        }
    }
}

pub struct Printer<'a> {
    pub info: &'a Info<'a>,
    pub tabs: u32,
    pub verbose: bool,
}

const TAB: &str = "  ";

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
