use crate::ast::*;

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
                ExprKind::Let(_, ty, _, _) => ty.for_each_type_rec(&mut f, stack),
                ExprKind::UnOp(UnOp::Cast(ty), _) => ty.for_each_type_rec(&mut f, stack),
                _ => {}
            }
        })
    }

    pub fn for_each_dim_expr<F: FnMut(&mut Expr, &mut Stack)>(&mut self, mut f: F) {
        self.for_each_type(|ty, stack| match &mut ty.kind {
            TypeKind::Array(_, shape) => match &mut shape.kind {
                ShapeKind::Ranked(dims) => dims.iter_mut().for_each(|dim| match &mut dim.kind {
                    DimKind::Expr(expr) => expr.for_each_expr_rec(&mut f, stack),
                    DimKind::Unknown => {}
                }),
                ShapeKind::Unranked => {}
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
            ExprKind::If(c, t, e) => {
                c.for_each_expr_rec(fun, stack);
                t.for_each_expr_rec(fun, stack);
                e.for_each_expr_rec(fun, stack);
            }
            ExprKind::Match(e, cases) => {
                e.for_each_expr_rec(fun, stack);
                cases
                    .iter_mut()
                    .for_each(|(_, e)| e.for_each_expr_rec(fun, stack));
            }
            ExprKind::Bif(kind) => match kind {
                Bif::Fmap(e) | Bif::Imap(_, e) => e.for_each_expr_rec(fun, stack),
                Bif::Fold(f, e) => {
                    f.for_each_expr_rec(fun, stack);
                    e.for_each_expr_rec(fun, stack);
                }
                Bif::Dataset(_) => {}
                Bif::Error => {}
            },
            ExprKind::Let(id, ty, v, b) => {
                v.for_each_expr_rec(fun, stack);
                stack.push((id.clone(), ty.clone()));
                b.for_each_expr_rec(fun, stack);
                stack.pop();
            }
            ExprKind::Call(_, ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ExprKind::Array(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ExprKind::Tuple(ps) => ps.iter_mut().for_each(|p| p.for_each_expr_rec(fun, stack)),
            ExprKind::Struct(fs) => fs
                .iter_mut()
                .for_each(|(_, v)| v.for_each_expr_rec(fun, stack)),
            ExprKind::Lit(_) => {}
            ExprKind::Var(_) => {}
            ExprKind::Error => {}
            ExprKind::BinOp(l, _, r) => {
                l.for_each_expr_rec(fun, stack);
                r.for_each_expr_rec(fun, stack);
            }
            ExprKind::UnOp(_, e) => e.for_each_expr_rec(fun, stack),
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
            TypeKind::Array(ty, _) => ty.for_each_type_rec(f, stack),
            TypeKind::Struct(fs) => fs
                .iter_mut()
                .for_each(|(_, ty)| ty.for_each_type_rec(f, stack)),
            TypeKind::Tuple(tys) => tys.iter_mut().for_each(|ty| ty.for_each_type_rec(f, stack)),
            TypeKind::Option(ty) => ty.for_each_type_rec(f, stack),
            TypeKind::I64
            | TypeKind::F64
            | TypeKind::I32
            | TypeKind::F32
            | TypeKind::Bool
            | TypeKind::Error
            | TypeKind::I8
            | TypeKind::I16
            | TypeKind::Null
            | TypeKind::String
            | TypeKind::Unknown => {}
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
