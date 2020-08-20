use BinOpKind::*;
use ExprKind::*;
use LitKind::*;
use ScalarKind::*;
use ShapeKind::*;
use TypeKind::*;
use UnOpKind::*;
use {
    crate::{ast::*, error::*, info::*},
    codespan::Span,
    ena::unify::{InPlace, UnificationTable, UnifyKey, UnifyValue},
};

pub type Context = UnificationTable<InPlace<TypeVar>>;

pub struct Typer {
    context: Context,
}

impl Typer {
    pub fn new() -> Typer {
        let context = Context::new();
        Typer { context }
    }
}

impl Typer {
    fn unify_var_var(&mut self, a: &Type, b: &Type, span: Span, errors: &mut Vec<CompilerError>) {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_var(a.var, b.var) {
            Ok(()) => self.context.commit(snapshot),
            Err((lhs, rhs)) => {
                errors.push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.context.rollback_to(snapshot)
            }
        }
    }

    fn unify_var_val(
        &mut self,
        a: &Type,
        b: &TypeKind,
        span: Span,
        errors: &mut Vec<CompilerError>,
    ) {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_value(a.var, b.clone()) {
            Ok(()) => self.context.commit(snapshot),
            Err((lhs, rhs)) => {
                errors.push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.context.rollback_to(snapshot)
            }
        }
    }

    fn fresh(&mut self) -> TypeVar {
        self.context.new_key(Unknown)
    }

    fn lookup(&mut self, var: TypeVar) -> TypeKind {
        self.context.probe_value(var)
    }
}

impl UnifyKey for TypeVar {
    type Value = TypeKind;

    fn index(&self) -> u32 {
        let TypeVar(id) = *self;
        id
    }

    fn from_index(id: u32) -> TypeVar {
        TypeVar(id)
    }

    fn tag() -> &'static str {
        "Type"
    }
}

impl UnifyValue for TypeKind {
    type Error = (Self, Self);

    fn unify_values(a: &Self, b: &Self) -> Result<Self, (Self, Self)> {
        match (a.clone(), b.clone()) {
            (Unknown, Unknown) => Ok(Unknown),
            (x, Unknown) | (Unknown, x) => Ok(x),
            (x, TypeErr) | (TypeErr, x) => Ok(x),
            (Array(ty1, sh1), Array(ty2, sh2)) => match (&sh1.kind, &sh2.kind) {
                (Ranked(r1), Ranked(r2)) if r1.len() != r2.len() => {
                    Err((Array(ty1.clone(), sh1), Array(ty2.clone(), sh2)))
                }
                (Ranked(_), Ranked(_)) => Ok(Array(ty1, sh1)),
                (Ranked(_), Unranked) | (Unranked, Ranked(_)) => Ok(Array(ty2, sh1)),
                (Unranked, Unranked) => Ok(Array(ty2, sh1)),
            },
            (Fun(mut args1, mut ret1), Fun(args2, ret2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.iter_mut().zip(args2.into_iter()) {
                    arg1.kind = TypeKind::unify_values(&arg1.kind, &arg2.kind)?;
                }
                ret1.kind = TypeKind::unify_values(&ret1.kind, &ret2.kind)?;
                Ok(Fun(args1, ret1))
            }
            (a, b) if a == b => Ok(a),
            (a, b) => Err((a, b)),
        }
    }
}

impl Script<'_> {
    pub fn infer(&mut self) {
        let typer = &mut Typer::new();
        let info = &mut self.info;
        self.ast.for_each_fun(|fun| fun.constrain(typer, info));
        self.ast
            .for_each_type(|ty| ty.var = typer.fresh(), &mut info.table);
        self.ast.for_each_expr(|expr| expr.constrain(typer, info));
        self.ast
            .for_each_type(|ty| ty.kind = typer.lookup(ty.var), &mut info.table);
    }
}

impl FunDef {
    fn constrain(&mut self, typer: &mut Typer, info: &mut Info) {
        let errors = &mut info.errors;
        if let Fun(arg_tys, ret_ty) = &info.table.get_decl(&self.id).ty.kind {
            typer.unify_var_var(&self.body.ty, ret_ty, self.body.span, errors);
            for (arg_ty, param) in arg_tys.iter().zip(&self.params) {
                let param_ty = &info.table.get_decl(&param).ty;
                typer.unify_var_var(param_ty, arg_ty, self.body.span, errors);
            }
        }
    }
}

impl Expr {
    fn constrain(&mut self, typer: &mut Typer, info: &mut Info) {
        let errors = &mut info.errors;
        match &self.kind {
            Let(id, v, b) => {
                let ty = &info.table.get_decl(id).ty;
                typer.unify_var_var(&v.ty, &ty, self.span, errors);
                typer.unify_var_var(&self.ty, &b.ty, self.span, errors);
            }
            Var(id) => {
                let ty = &info.table.get_decl(id).ty;
                typer.unify_var_var(&self.ty, &ty, self.span, errors);
            }
            Lit(l) => {
                let kind = match l {
                    LitI8(_) => Scalar(I8),
                    LitI16(_) => Scalar(I16),
                    LitI32(_) => Scalar(I32),
                    LitI64(_) => Scalar(I64),
                    LitF32(_) => Scalar(F32),
                    LitF64(_) => Scalar(F64),
                    LitBool(_) => Scalar(Bool),
                    LitTime(_) => todo!(),
                    LitErr => return,
                };
                typer.unify_var_val(&self.ty, &kind, self.span, errors);
            }
            ConsArray(args) => {
                let mut elem_ty = Type::new();
                elem_ty.var = typer.fresh();
                let size = args.len() as i32;
                args.iter()
                    .for_each(|e| typer.unify_var_var(&elem_ty, &e.ty, self.span, errors));
                let kind = Array(Box::new(elem_ty), Shape::simple(size, self.span));
                typer.unify_var_val(&self.ty, &kind, self.span, errors);
            }
            ConsStruct(fields) => {
                let kind = Struct(
                    fields
                        .iter()
                        .map(|(sym, e)| (*sym, e.ty.clone()))
                        .collect::<Vec<_>>(),
                );
                typer.unify_var_val(&self.ty, &kind, self.span, errors);
            }
            ConsTuple(args) => {
                let kind = Tuple(args.iter().map(|arg| arg.ty.clone()).collect());
                typer.unify_var_val(&self.ty, &kind, self.span, errors);
            }
            BinOp(l, kind, r) => {
                typer.unify_var_var(&l.ty, &r.ty, self.span, errors);
                match kind {
                    Add | Div | Mul | Sub => {
                        typer.unify_var_var(&self.ty, &r.ty, self.span, errors)
                    }
                    Eq => typer.unify_var_val(&self.ty, &Scalar(Bool), self.span, errors),
                    BinOpErr => {}
                }
            }
            UnOp(kind, e) => {
                match kind {
                    Not => typer.unify_var_val(&e.ty, &Scalar(Bool), e.span, errors),
                    Cast(ty) => typer.unify_var_val(&e.ty, &ty.kind, e.span, errors),
                    MethodCall(_, _) => return, // TODO
                    Project(_) => return,
                    Access(_) => return,
                    UnOpErr => return,
                }
                typer.unify_var_var(&self.ty, &e.ty, e.span, errors);
            }
            If(c, t, e) => {
                typer.unify_var_val(&c.ty, &Scalar(Bool), c.span, errors);
                typer.unify_var_var(&t.ty, &e.ty, e.span, errors);
                typer.unify_var_var(&t.ty, &self.ty, e.span, errors);
            }
            Closure(..) => todo!(),
            Match(_, _) => {}
            FunCall(id, args) => {
                let ty = &info.table.get_decl(id).ty;
                let fun = Fun(
                    args.iter().map(|arg| arg.ty.clone()).collect(),
                    Box::new(self.ty.clone()),
                );
                typer.unify_var_val(&ty, &fun, self.span, errors);
            }
            ExprErr => {}
        }
    }
}
