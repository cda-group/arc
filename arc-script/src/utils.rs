use crate::ast::*;
use crate::info::Info;
use crate::symbols::SymbolTable;
use DimKind::*;
use ExprKind::*;
use ShapeKind::*;
use TypeKind::*;
use UnOpKind::*;

impl SyntaxTree {
    pub fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr(f));
        self.body.for_each_expr(f);
    }

    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|fundef| f(fundef));
    }

    pub fn for_each_decl<F: FnMut(&mut Decl)>(&mut self, ref mut f: F, table: &mut SymbolTable) {
        table.decls.iter_mut().for_each(f);
    }

    pub fn for_each_type<F: FnMut(&mut Type)>(&mut self, mut f: F, table: &mut SymbolTable) {
        self.for_each_decl(|decl| decl.ty.for_each_type(&mut f), table);
        self.for_each_expr(|e| {
            e.ty.for_each_type_rec(&mut f);
            if let UnOp(Cast(ty), _) = &mut e.kind {
                ty.for_each_type_rec(&mut f)
            }
        })
    }

    pub fn for_each_dim_expr<F: FnMut(&mut Expr)>(&mut self, mut f: F, table: &mut SymbolTable) {
        self.for_each_type(
            |ty| {
                if let Array(_, shape) = &mut ty.kind {
                    match &mut shape.kind {
                        Ranked(dims) => dims.iter_mut().for_each(|dim| match &mut dim.kind {
                            Symbolic(expr) => expr.for_each_expr(&mut f),
                            Hole => {}
                        }),
                        Unranked => {}
                    }
                }
            },
            table,
        )
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
            Let(_, v, b) => {
                v.for_each_expr(f);
                b.for_each_expr(f);
            }
            FunCall(_, ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr(f)),
            ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.for_each_expr(f)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr(f);
                r.for_each_expr(f);
            }
            UnOp(_, e) => e.for_each_expr(f),
            Closure(..) => todo!(),
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
