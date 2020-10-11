use {
    crate::{error::*, info::*, prelude::*, symbols::*},
    codespan::Span,
    derive_more::Constructor,
    ena::unify::{InPlace, UnifyKey, UnifyValue},
};

pub type UnificationTable = ena::unify::UnificationTable<InPlace<TypeVar>>;

pub struct Typer {
    table: UnificationTable,
}

/// Lumps together all values which are needed during typeinference for brevity.
#[derive(Constructor)]
struct Context<'i> {
    typer: &'i mut Typer,
    span: Span,
    errors: &'i mut Vec<CompilerError>,
    table: &'i SymbolTable,
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
    /// Returns a new Typer
    pub fn new() -> Typer {
        let table = UnificationTable::new();
        Typer { table }
    }
    /// Returns a fresh type variable which is unified with the given type `ty`.
    pub fn intern<T: Into<Type>>(&mut self, ty: T) -> TypeVar {
        self.table.new_key(ty.into())
    }

    /// Returns a fresh type variable.
    pub fn fresh(&mut self) -> TypeVar {
        self.table.new_key(Type::new())
    }

    /// Returns the type which `tv` is unified with.
    pub fn lookup(&mut self, tv: TypeVar) -> Type {
        self.table.probe_value(tv)
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

impl<'i> Context<'i> {
    /// Unifies two type variables `a` and `b`.
    fn unify_var_var(&mut self, a: TypeVar, b: TypeVar) {
        let snapshot = self.typer.table.snapshot();
        match self.typer.table.unify_var_var(a, b) {
            Ok(()) => self.typer.table.commit(snapshot),
            Err((lhs, rhs)) => {
                let span = self.span;
                self.errors
                    .push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.typer.table.rollback_to(snapshot)
            }
        }
    }

    /// Unifies a type variable `a` with a type `b`.
    fn unify_var_val<T>(&mut self, a: TypeVar, b: T)
    where
        T: Into<Type>,
    {
        let snapshot = self.typer.table.snapshot();
        match self.typer.table.unify_var_value(a, b.into()) {
            Ok(()) => self.typer.table.commit(snapshot),
            Err((lhs, rhs)) => {
                let span = self.span;
                self.errors
                    .push(CompilerError::TypeMismatch { lhs, rhs, span });
                self.typer.table.rollback_to(snapshot)
            }
        }
    }

    /// Unifies two polytypes, i.e., types which may contain type variables
    fn unify(&mut self, tv1: TypeVar, tv2: TypeVar) {
        let ty1 = self.typer.lookup(tv1);
        let ty2 = self.typer.lookup(tv2);
        match (&ty1.kind, &ty2.kind) {
            (Unknown, Unknown) => self.unify_var_var(tv1, tv2),
            (Unknown, _) => self.unify_var_val(tv1, ty2),
            (_, Unknown) => self.unify_var_val(tv2, ty1),
            (Array(tv1, sh1), Array(tv2, sh2)) if sh1.dims.len() == sh2.dims.len() => {
                self.unify(*tv1, *tv2);
            }
            (Fun(args1, ret1), Fun(args2, ret2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.into_iter().zip(args2.into_iter()) {
                    self.unify(*arg1, *arg2);
                }
                self.unify(*ret1, *ret2);
            }
            (Tuple(args1), Tuple(args2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.into_iter().zip(args2.into_iter()) {
                    self.unify(*arg1, *arg2);
                }
            }
            (Struct(map1), Struct(map2)) => {
                for (field1, tv1) in map1.into_iter() {
                    if let Some(tv2) = map2.get(field1) {
                        self.unify(*tv1, *tv2);
                    }
                }
            }
            (Enum(map1), Enum(map2)) => {
                for (variant1, tv1) in map1.into_iter() {
                    if let Some(tv2) = map2.get(variant1) {
                        self.unify(*tv1, *tv2);
                    }
                }
            }
            (Stream(tv1), Stream(tv2)) => self.unify(*tv1, *tv2),
            (Optional(tv1), Optional(tv2)) => self.unify(*tv1, *tv2),
            // This seems a bit out of place, but it is needed to ensure that monotypes unify
            _ => self.unify_var_var(tv1, tv2),
        }
    }
}

impl Script<'_> {
    /// Infers the types of all type variables in a Script.
    pub fn infer(&mut self) {
        let Info {
            table,
            typer,
            errors,
            ..
        } = &mut self.info;
        let typer = typer.get_mut();
        self.ast.for_each_expr(|expr| {
            let ctx = &mut Context::new(typer, expr.span, errors, table);
            expr.constrain(ctx)
        });
        self.ast.fundefs.iter_mut().for_each(|(id, fundef)| {
            let ctx = &mut Context::new(typer, fundef.body.span, errors, table);
            (id, fundef).constrain(ctx);
        });
    }
}

trait Constrain<'i> {
    fn constrain(&mut self, context: &mut Context<'i>);
}

impl<'i> Constrain<'i> for (&Ident, &mut FunDef) {
    /// Constrains the types of a function based on its signature and body.
    fn constrain(&mut self, ctx: &mut Context<'i>) {
        let (id, fundef) = self;
        let tv = ctx.table.get_decl(&id).tv;
        let ty = ctx.typer.lookup(tv);
        if let Fun(_, ret_tv) = ty.kind {
            ctx.unify(fundef.body.tv, ret_tv)
        }
    }
}

impl<'i> Constrain<'i> for Expr {
    /// Constrains an expression based on its subexpressions.
    fn constrain(&mut self, ctx: &mut Context<'i>) {
        match &self.kind {
            Let(id, v) => {
                let tv = ctx.table.get_decl(id).tv;
                ctx.unify(v.tv, tv);
                ctx.unify_var_val(self.tv, Unit);
            }
            Var(id) => {
                let tv = ctx.table.get_decl(id).tv;
                ctx.unify(self.tv, tv);
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
                    LitUnit => Unit,
                    LitTime(_) => todo!(),
                    LitErr => return,
                };
                ctx.unify_var_val(self.tv, kind);
            }
            ConsArray(args) => {
                let elem_tv = ctx.typer.fresh();
                let dim = Dim::from(DimVal(args.len() as i32));
                args.iter().for_each(|e| ctx.unify(elem_tv, e.tv));
                let shape = Shape::from(vec![dim]);
                ctx.unify_var_val(self.tv, Array(elem_tv, shape));
            }
            ConsStruct(fields) => {
                let fields = fields
                    .iter()
                    .map(|(field, arg)| (*field, arg.tv))
                    .collect::<VecMap<_, _>>();
                ctx.unify_var_val(self.tv, Struct(fields));
            }
            ConsEnum(variants) => {
                let variants = variants
                    .iter()
                    .map(|(variant, arg)| (*variant, arg.tv))
                    .collect::<VecMap<_, _>>();
                ctx.unify_var_val(self.tv, Enum(variants));
            }
            ConsTuple(args) => {
                let tvs = args.iter().map(|arg| arg.tv).collect();
                ctx.unify_var_val(self.tv, Tuple(tvs));
            }
            For(..) => todo!(),
            BinOp(l, kind, r) => match kind {
                Add | Div | Mul | Sub => {
                    ctx.unify(l.tv, r.tv);
                    ctx.unify(self.tv, r.tv)
                }
                Pow => {
                    ctx.unify(self.tv, l.tv);
                    if let Scalar(kind) = ctx.typer.lookup(l.tv).kind {
                        match kind {
                            I8 | I16 | I32 | I64 => ctx.unify_var_val(r.tv, I32),
                            F32 => ctx.unify_var_val(r.tv, F32),
                            F64 => ctx.unify_var_val(r.tv, F64),
                            _ => {}
                        }
                    }
                    if let Scalar(kind) = ctx.typer.lookup(r.tv).kind {
                        match kind {
                            F32 => ctx.unify_var_val(l.tv, F32),
                            F64 => ctx.unify_var_val(l.tv, F64),
                            _ => {}
                        }
                    }
                }
                Equ | Neq | Gt | Lt | Geq | Leq => {
                    ctx.unify(l.tv, r.tv);
                    ctx.unify_var_val(self.tv, Bool)
                }
                Or | And => {
                    ctx.unify(self.tv, l.tv);
                    ctx.unify(self.tv, r.tv);
                    ctx.unify_var_val(self.tv, Bool)
                }
                Pipe => todo!(),
                Seq => ctx.unify(self.tv, r.tv),
                BinOpErr => return,
            },
            UnOp(kind, e) => match kind {
                Not => {
                    ctx.unify(self.tv, e.tv);
                    ctx.unify_var_val(e.tv, Bool);
                }
                Neg => ctx.unify(self.tv, e.tv),
                Cast(tv) => ctx.unify(e.tv, *tv),
                Project(idx) => {
                    if let Tuple(tvs) = ctx.typer.lookup(e.tv).kind {
                        if let Some(tv) = tvs.get(idx.0) {
                            ctx.unify_var_var(self.tv, *tv);
                        } else {
                            ctx.errors
                                .push(CompilerError::OutOfBoundsProject { span: self.span })
                        }
                    }
                }
                Access(sym) => {
                    if let Struct(fields) = ctx.typer.lookup(e.tv).kind {
                        if let Some(tv) = fields.get(sym) {
                            ctx.unify_var_var(self.tv, *tv);
                        } else {
                            ctx.errors
                                .push(CompilerError::FieldNotFound { span: self.span })
                        }
                    }
                }
                Call(args) => {
                    let params = args.iter().map(|arg| arg.tv).collect();
                    let tv2 = ctx.typer.intern(Fun(params, self.tv));
                    ctx.unify(e.tv, tv2);
                }
                Emit => {
                    // TODO: Ensure that arg is a sink (enum variant)
                    ctx.unify_var_val(self.tv, Unit);
                }
                UnOpErr => return,
            },
            If(c, t, e) => {
                ctx.unify_var_val(c.tv, Bool);
                ctx.unify(t.tv, e.tv);
                ctx.unify(t.tv, self.tv);
            }
            Closure(params, body) => {
                let params = params
                    .iter()
                    .map(|param| ctx.table.get_decl(param).tv)
                    .collect();
                let tv = ctx.typer.intern(Fun(params, body.tv));
                ctx.unify(self.tv, tv)
            }
            Match(_, _) => {}
            Sink(_) => todo!(),
            Source(_) => todo!(),
            Loop(_, _) => todo!(),
            ExprErr => return,
        }
    }
}
