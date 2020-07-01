use {
    crate::{ast::*, error::*, utils::*},
    codespan::Span,
    ena::unify::{InPlace, UnificationTable, UnifyKey, UnifyValue},
};

pub type Context = UnificationTable<InPlace<TypeVar>>;

impl Typer {
    pub fn new() -> Typer {
        let context = Context::new();
        let errors = Vec::new();
        Typer { context, errors }
    }
}

pub struct Typer {
    context: Context,
    errors: Vec<CompilerError>,
}

fn type_mismatch<'i>((lhs, rhs): (TypeKind, TypeKind), span: Span) -> CompilerError {
    CompilerError::TypeMismatch {
        lhs: lhs.to_string(),
        rhs: rhs.to_string(),
        span,
    }
}

impl Typer {
    fn unify_var_var(&mut self, a: &Type, b: &Type, span: Span) {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_var(a.var, b.var) {
            Ok(()) => self.context.commit(snapshot),
            Err(err) => {
                self.errors.push(type_mismatch(err, span));
                self.context.rollback_to(snapshot)
            }
        }
    }

    fn unify_var_val(&mut self, a: &Type, b: &TypeKind, span: Span) {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_value(a.var, b.clone()) {
            Ok(()) => self.context.commit(snapshot),
            Err(err) => {
                self.errors.push(type_mismatch(err, span));
                self.context.rollback_to(snapshot)
            }
        }
    }

    fn fresh(&mut self) -> TypeVar { self.context.new_key(TypeKind::Unknown) }

    fn lookup(&mut self, var: TypeVar) -> TypeKind { self.context.probe_value(var) }

    pub fn errors(self) -> Vec<CompilerError> { self.errors }
}

impl UnifyKey for TypeVar {
    type Value = TypeKind;

    fn index(&self) -> u32 {
        let TypeVar(id) = *self;
        id
    }

    fn from_index(id: u32) -> TypeVar { TypeVar(id) }

    fn tag() -> &'static str { "Type" }
}

impl UnifyValue for TypeKind {
    type Error = (TypeKind, TypeKind);

    fn unify_values(a: &TypeKind, b: &TypeKind) -> Result<TypeKind, (TypeKind, TypeKind)> {
        use ShapeKind::*;
        use TypeKind::*;
        match (a.clone(), b.clone()) {
            (Unknown, Unknown) => Ok(Unknown),
            (x, Unknown) | (Unknown, x) => Ok(x),
            (x, Error) | (Error, x) => Ok(x),
            (Array(ty1, sh1), Array(ty2, sh2)) => match (&sh1.kind, &sh2.kind) {
                (Ranked(r1), Ranked(r2)) if r1.len() != r2.len() => {
                    Err((Array(ty1.clone(), sh1), Array(ty2.clone(), sh2)))
                }
                (Ranked(_), Ranked(_)) => Ok(Array(ty1, sh1)),
                (Ranked(_), Unranked) | (Unranked, Ranked(_)) => Ok(Array(ty2, sh1)),
                (Unranked, Unranked) => Ok(Array(ty2, sh1)),
            },
            (a, b) if a == b => Ok(a),
            (a, b) => Err((a, b)),
        }
    }
}

impl Expr {
    pub fn infer(&mut self, typer: &mut Typer) {
        self.for_each_type(|ty, _| ty.var = typer.fresh());
        self.for_each_expr(|expr, stack| expr.constrain(typer, stack));
        self.for_each_type(|ty, _| ty.kind = typer.lookup(ty.var));
    }

    fn constrain(&mut self, typer: &mut Typer, stack: &mut Stack) {
        match &self.kind {
            ExprKind::Let(_, ty, v, b) => {
                typer.unify_var_var(&v.ty, &ty, self.span);
                typer.unify_var_var(&self.ty, &b.ty, self.span);
            }
            ExprKind::Var(id) => match id.lookup_with_scope(stack) {
                Some((ty, _, _)) => typer.unify_var_var(&self.ty, &ty, self.span),
                None => typer.errors.push(CompilerError::VarNotFound {
                    name: id.name.clone(),
                    span: self.span,
                }),
            },
            ExprKind::Lit(l) => {
                let kind = match l {
                    Lit::I32(_) => TypeKind::I32,
                    Lit::I64(_) => TypeKind::I64,
                    Lit::F32(_) => TypeKind::F32,
                    Lit::F64(_) => TypeKind::F64,
                    Lit::Bool(_) => TypeKind::Bool,
                    Lit::Array(args) => {
                        let mut elem_ty = Type::new();
                        elem_ty.var = typer.fresh();
                        let size = args.len() as i32;
                        args.iter()
                            .for_each(|e| typer.unify_var_var(&elem_ty, &e.ty, self.span));
                        TypeKind::Array(Box::new(elem_ty), Shape::simple(size, self.span))
                    }
                    Lit::Struct(fields) => TypeKind::Struct(
                        fields
                            .iter()
                            .map(|(id, e)| (id.clone(), e.ty.clone()))
                            .collect::<Vec<(Ident, Type)>>(),
                    ),
                    Lit::Error => return,
                };
                typer.unify_var_val(&self.ty, &kind, self.span);
            }
            ExprKind::BinOp(l, op, r) => {
                typer.unify_var_var(&l.ty, &r.ty, self.span);
                match op {
                    BinOp::Add | BinOp::Div | BinOp::Mul | BinOp::Sub => {
                        typer.unify_var_var(&self.ty, &r.ty, self.span);
                    }
                    BinOp::Eq => {
                        typer.unify_var_val(&self.ty, &TypeKind::Bool, self.span);
                    }
                    BinOp::Error => {}
                }
            }
            ExprKind::UnOp(op, e) => {
                match op {
                    UnOp::Not => typer.unify_var_val(&e.ty, &TypeKind::Bool, e.span),
                    UnOp::Cast(ty) => typer.unify_var_val(&e.ty, &ty.kind, e.span),
                    UnOp::Call(_, _) => return, // TODO
                    UnOp::Project(_) => return,
                    UnOp::Access(_) => return,
                    UnOp::Error => return,
                }
                typer.unify_var_var(&self.ty, &e.ty, e.span);
            }
            ExprKind::If(c, t, e) => {
                typer.unify_var_val(&c.ty, &TypeKind::Bool, c.span);
                typer.unify_var_var(&t.ty, &e.ty, e.span);
                typer.unify_var_var(&t.ty, &self.ty, e.span);
            }
            ExprKind::Match(_, _) => {}
            ExprKind::Bif(_) => unimplemented!(),
            ExprKind::Call(_, _) => unimplemented!(),
            ExprKind::Error => {}
        }
    }
}
