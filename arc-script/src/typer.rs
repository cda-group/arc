use BinOpKind::*;
use DimKind::*;
use ExprKind::*;
use LitKind::*;
use ScalarKind::*;
use TypeKind::*;
use UnOpKind::*;
use {
    crate::{ast::*, error::*, info::*, symbols::*},
    codespan::Span,
    ena::unify::{InPlace, UnificationTable, UnifyKey, UnifyValue},
};

pub type TypingContext = UnificationTable<InPlace<TypeVar>>;

pub struct Typer {
    context: TypingContext,
}

/// Suppose we have the following arc-script code:
/// ---------------------------------------------
/// fun max(a: i32, b: i32) -> i32
///   c = a > b
///   if c then a else b
///
/// a = 5
/// b = 7
/// ---------------------------------------------
/// With type variables (some omitted):
/// ---------------------------------------------
/// fun max(a: 1→i32, b: 2→i32) -> 3→i32
///   c = a > b: 3→?
///   if c then a else b: 4→?
///
/// a = 5: 5→?
/// b = 7: 6→?
/// max(a,b): 7→?
/// ---------------------------------------------
/// With type constraints:
/// ---------------------------------------------
/// fun max(a: 1→i32, b: 2→i32) -> 3→i32
///   c = a > b: 3→bool
///   if c then a else b: 4→?
///
/// a = 5: 5→i32
/// b = 7: 6→i32
/// max(a,b): 7→(Fun(8→i32, 9→i32) -> 10→i32)
/// ---------------------------------------------
///
/// unify_var_val(7, Fun(8→i32, 9→i32) -> 10→8)
/// unify_var_val(7, Fun(1→i32, 2→i32) -> 3→i32)
/// unify_values(
///     Fun(8→i32, 9→i32) -> 10→8,
///     Fun(1→i32, 2→i32) -> 3→i32
/// )
///
/// unify_values(
///     Fun(8, 9) -> 10,
///     Fun(1, 2) -> 3
/// )
///
/// unify_var_value(1, Fun(2,3))
///   1 → Fun(2,3)

impl Typer {
    pub fn new() -> Typer {
        let context = TypingContext::new();
        Typer { context }
    }
}

impl Typer {
    fn unify_var_var(
        &mut self,
        a: TypeVar,
        b: TypeVar,
        span: Span,
        errors: &mut Vec<CompilerError>,
    ) {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_var(a, b) {
            Ok(()) => self.context.commit(snapshot),
            Err((lhs, rhs)) => {
                errors.push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.context.rollback_to(snapshot)
            }
        }
    }

    fn unify_var_val<T>(&mut self, a: TypeVar, b: T, span: Span, errors: &mut Vec<CompilerError>)
    where
        T: Into<Type>,
    {
        let snapshot = self.context.snapshot();
        match self.context.unify_var_value(a, b.into()) {
            Ok(()) => self.context.commit(snapshot),
            Err((lhs, rhs)) => {
                errors.push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.context.rollback_to(snapshot)
            }
        }
    }

    pub fn intern(&mut self, ty: Type) -> TypeVar {
        self.context.new_key(ty)
    }

    pub fn fresh(&mut self) -> TypeVar {
        self.context.new_key(Type::new())
    }

    pub fn lookup(&mut self, var: TypeVar) -> Type {
        self.context.probe_value(var)
    }
}

impl UnifyKey for TypeVar {
    type Value = Type;

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

impl UnifyValue for Type {
    type Error = (Self, Self);

    /// Unifies two monotypes, i.e., types which contain no type variables
    /// Some logic is duplicated between mono- and poly-type unification which is
    /// maybe not ideal. The reason for why unify_values cannot do poly-type
    /// unification is it's not possible to access the Typer from within the function,
    /// so we cannot do recursive lookups for values of type variables. For this reason,
    /// all unification needs to be "top-level".
    fn unify_values(ty1: &Self, ty2: &Self) -> Result<Self, Self::Error> {
        if let (Array(_, sh1), Array(_, sh2)) = (&ty1.kind, &ty2.kind) {
            match (&sh1.dims, &sh2.dims) {
                (d1, _) if d1.iter().all(|dim| dim.is_val()) => Ok(ty1.clone()),
                (_, d2) if d2.iter().all(|dim| dim.is_val()) => Ok(ty2.clone()),
                _ => Err((ty1.clone(), ty2.clone())),
            }
        } else {
            match (&ty1.kind, &ty2.kind) {
                (Unknown, _) | (TypeErr, _) => Ok(ty2.clone()),
                (_, Unknown) | (_, TypeErr) => Ok(ty1.clone()),
                (a, b) => {
                    if a == b {
                        Ok(ty1.clone())
                    } else {
                        Err((ty1.clone(), ty2.clone()))
                    }
                }
            }
        }
    }
}

impl Typer {
    /// Unifies two polytypes, i.e., types which may contain type variables
    fn unify(&mut self, tv1: TypeVar, tv2: TypeVar, span: Span, errors: &mut Vec<CompilerError>) {
        let ty1 = self.lookup(tv1);
        let ty2 = self.lookup(tv2);
        match (&ty1.kind, &ty2.kind) {
            (Unknown, Unknown) => self.unify_var_var(tv1, tv2, span, errors),
            (Unknown, _) => self.unify_var_val(tv1, ty2, span, errors),
            (_, Unknown) => self.unify_var_val(tv2, ty1, span, errors),
            (Array(tv1, sh1), Array(tv2, sh2)) if sh1.dims.len() == sh2.dims.len() => {
                self.unify(*tv1, *tv2, span, errors);
            }
            (Fun(args1, ret1), Fun(args2, ret2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.into_iter().zip(args2.into_iter()) {
                    self.unify(*arg1, *arg2, span, errors);
                }
                self.unify(*ret1, *ret2, span, errors);
            }
            // This seems a bit out of place, but it is needed to ensure that monotypes unify
            _ => self.unify_var_var(tv1, tv2, span, errors),
        }
    }
}

impl Script<'_> {
    pub fn infer(&mut self) {
        let Info {
            table,
            typer,
            errors,
            ..
        } = &mut self.info;
        let typer = typer.get_mut();
        self.ast
            .for_each_expr(|expr| expr.constrain(typer, errors, table));
        self.ast
            .for_each_fun(|fun| fun.constrain(typer, errors, table));
    }
}

impl FunDef {
    fn constrain(
        &mut self,
        typer: &mut Typer,
        errors: &mut Vec<CompilerError>,
        table: &mut SymbolTable,
    ) {
        let tv = table.get_decl(&self.id).tv;
        let ty = typer.context.probe_value(tv);
        if let Fun(_, ret_tv) = ty.kind {
            typer.unify(self.body.tv, ret_tv, self.body.span, errors)
        }
    }
}

impl Expr {
    fn constrain(
        &mut self,
        typer: &mut Typer,
        errors: &mut Vec<CompilerError>,
        table: &mut SymbolTable,
    ) {
        match &self.kind {
            Let(id, v, b) => {
                let tv = table.get_decl(id).tv;
                typer.unify(v.tv, tv, self.span, errors);
                typer.unify(self.tv, b.tv, self.span, errors);
            }
            Var(id) => {
                let tv = table.get_decl(id).tv;
                typer.unify(self.tv, tv, self.span, errors);
            }
            Lit(l) => {
                let kind = match l {
                    LitI8(_) => I8,
                    LitI16(_) => I16,
                    LitI32(_) => I32,
                    LitI64(_) => I64,
                    LitF32(_) => F32,
                    LitF64(_) => F64,
                    LitBool(_) => Bool,
                    LitTime(_) => todo!(),
                    LitErr => return,
                };
                typer.unify_var_val(self.tv, Scalar(kind), self.span, errors);
            }
            ConsArray(args) => {
                let elem_tv = typer.fresh();
                let dim = Dim::from(DimVal(args.len() as i32));
                args.iter()
                    .for_each(|e| typer.unify(elem_tv, e.tv, self.span, errors));
                let shape = Shape::from(vec![dim]);
                typer.unify_var_val(self.tv, Array(elem_tv, shape), self.span, errors);
            }
            ConsStruct(fields) => {
                let fields = fields
                    .iter()
                    .map(|(sym, e)| (*sym, e.tv))
                    .collect::<Vec<_>>();
                typer.unify_var_val(self.tv, Struct(fields), self.span, errors);
            }
            ConsTuple(args) => {
                let tvs = args.iter().map(|arg| arg.tv).collect();
                typer.unify_var_val(self.tv, Tuple(tvs), self.span, errors);
            }
            BinOp(l, kind, r) => {
                typer.unify(l.tv, r.tv, self.span, errors);
                match kind {
                    Add | Div | Mul | Sub => typer.unify(self.tv, r.tv, self.span, errors),
                    Eq | Neq | Gt | Lt | Geq | Leq => {
                        typer.unify_var_val(self.tv, Scalar(Bool), self.span, errors)
                    }
                    BinOpErr => {}
                }
            }
            UnOp(kind, e) => {
                match kind {
                    Not => typer.unify_var_val(e.tv, Scalar(Bool), e.span, errors),
                    Cast(tv) => typer.unify(e.tv, *tv, e.span, errors),
                    MethodCall(_, _) => return, // TODO
                    Project(_) => return,
                    Access(_) => return,
                    UnOpErr => return,
                }
                typer.unify(self.tv, e.tv, e.span, errors);
            }
            If(c, t, e) => {
                typer.unify_var_val(c.tv, Scalar(Bool), c.span, errors);
                typer.unify(t.tv, e.tv, e.span, errors);
                typer.unify(t.tv, self.tv, e.span, errors);
            }
            Closure(..) => todo!(),
            Match(_, _) => {}
            FunCall(id, args) => {
                let tv = table.get_decl(id).tv;
                let fun = Fun(args.iter().map(|arg| arg.tv).collect(), self.tv);
                typer.unify_var_val(tv, fun, self.span, errors);
            }
            ExprErr => {}
        }
    }
}
