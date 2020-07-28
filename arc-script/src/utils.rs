use crate::ast::*;
use crate::ExprKind::*;
use crate::ShapeKind::*;
use crate::UnOpKind::*;
use crate::TypeKind::*;
use crate::DimKind::*;

pub type Stack = Vec<(Ident, Type)>;

impl Expr {
    pub fn for_each_expr<F: FnMut(&mut Expr, &mut Stack)>(&mut self, mut fun: F) {
        let mut stack = Vec::new();
        self.for_each_expr_rec(&mut fun, &mut stack);
    }

    pub fn for_each_type<F: FnMut(&mut Type, &mut Stack)>(&mut self, mut f: F) {
        self.for_each_expr(|e, stack| {
            e.ty.for_each_type_rec(&mut f, stack);
            match &mut e.kind {
                Let(_, ty, _, _) => ty.for_each_type_rec(&mut f, stack),
                UnOp(Cast(ty), _) => ty.for_each_type_rec(&mut f, stack),
                _ => {}
            }
        })
    }

    pub fn for_each_dim_expr<F: FnMut(&mut Expr, &mut Stack)>(&mut self, mut f: F) {
        self.for_each_type(|ty, stack| match &mut ty.kind {
            Array(_, shape) => match &mut shape.kind {
                Ranked(dims) => dims.iter_mut().for_each(|dim| match &mut dim.kind {
                    Symbolic(expr) => expr.for_each_expr_rec(&mut f, stack),
                    Hole => {}
                }),
                Unranked => {}
            },
            _ => {}
        })
    }

    fn for_each_expr_rec<F: FnMut(&mut Expr, &mut Stack)>(
        &mut self,
        fun: &mut F,
        stack: &mut Stack,
    ) {
        fun(self, stack);
        match &mut self.kind {
            If(c, t, e) => {
                c.for_each_expr_rec(fun, stack);
                t.for_each_expr_rec(fun, stack);
                e.for_each_expr_rec(fun, stack);
            }
            Match(e, cases) => {
                e.for_each_expr_rec(fun, stack);
                cases
                    .iter_mut()
                    .for_each(|(_, e)| e.for_each_expr_rec(fun, stack));
            }
            Let(id, ty, v, b) => {
                v.for_each_expr_rec(fun, stack);
                stack.push((id.clone(), ty.clone()));
                b.for_each_expr_rec(fun, stack);
                stack.pop();
            }
            FunCall(_, ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ConsArray(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ConsTuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ConsStruct(fs) => fs
                .iter_mut()
                .for_each(|(_, v)| v.for_each_expr_rec(fun, stack)),
            Lit(_) => {}
            Var(_) => {}
            BinOp(l, _, r) => {
                l.for_each_expr_rec(fun, stack);
                r.for_each_expr_rec(fun, stack);
            }
            UnOp(_, e) => e.for_each_expr_rec(fun, stack),
            ExprErr => {}
        }
    }
}
impl Type {
    pub fn for_each_type<F: FnMut(&mut Type, &mut Stack)>(&mut self, mut f: F) {
        let mut stack = Vec::new();
        self.for_each_type_rec(&mut f, &mut stack);
    }

    fn for_each_type_rec<F: FnMut(&mut Type, &mut Stack)>(&mut self, f: &mut F, stack: &mut Stack) {
        f(self, stack);
        match &mut self.kind {
            Array(ty, _) => ty.for_each_type_rec(f, stack),
            Struct(fs) => fs
                .iter_mut()
                .for_each(|(_, ty)| ty.for_each_type_rec(f, stack)),
            Tuple(tys) => tys.iter_mut().for_each(|ty| ty.for_each_type_rec(f, stack)),
            Optional(ty) => ty.for_each_type_rec(f, stack),
            Scalar(_) => {},
            Unknown => {},
            _ => todo!()
        }
    }
}

impl Ident {
    pub fn lookup_with_name(&self, stack: &Stack) -> Option<(Type, Scope, Option<Uid>)> {
        stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(scope, (id, ty))| match id.name == self.name {
                true => Some((ty.clone(), scope, id.uid.clone())),
                false => None,
            })
    }

    pub fn lookup_with_uid(&self, stack: &Stack) -> Option<(Type, Scope, Option<Uid>)> {
        self.uid.and_then(|ref uid1| {
            stack
                .iter()
                .enumerate()
                .rev()
                .find_map(|(scope, (id, ty))| {
                    id.uid
                        .filter(|uid2| uid2 == uid1)
                        .map(|_| (ty.clone(), scope, id.uid.clone()))
                })
        })
    }

    pub fn lookup_with_scope(&self, stack: &Stack) -> Option<(Type, Scope, Option<Uid>)> {
        let scope = self
            .scope
            .expect("[ICE]: Tried to access unassigned scope.");
        stack.get(scope).map(|(id, ty)| (ty.clone(), scope, id.uid))
    }
}

const TAB: &'static str = "  ";
pub fn indent(indent: u32) -> String {
    format!(
        "\n{}",
        (0..indent).into_iter().map(|_| TAB).collect::<String>()
    )
}

pub fn merge<T>(mut a: Vec<T>, mut b: Vec<T>) -> Vec<T> {
    a.append(&mut b);
    a
}
