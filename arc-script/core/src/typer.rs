use {
    crate::{error::*, info::*, prelude::*, symbols::*},
    ena::unify::{InPlace, NoError, UnifyKey, UnifyValue},
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

impl Default for Typer {
    fn default() -> Self {
        let table = UnificationTable::new();
        Typer { table }
    }
}

impl Typer {
    /// Returns a new Typer
    pub fn new() -> Typer {
        Self::default()
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
    type Error = NoError;

    /// Unifies an unknown type with an inferred type.
    /// except at the top-level.
    fn unify_values(ty1: &Self, ty2: &Self) -> Result<Self, Self::Error> {
        match (&ty1.kind, &ty2.kind) {
            (Unknown, _) | (TypeErr, _) => Ok(ty2.clone()),
            (_, Unknown) | (_, TypeErr) => Ok(ty1.clone()),
            _ => Ok(ty1.clone()),
        }
    }
}

trait Unify<A, B> {
    fn union(&mut self, a: A, b: B);
    fn unify(&mut self, a: A, b: B);
}

/// Unifies a type variable `a` with a type `b`.
impl<T: Into<Type>> Unify<TypeVar, T> for Context<'_> {
    fn union(&mut self, a: TypeVar, b: T) {
        self.typer.table.union_value(a, b.into());
    }
    fn unify(&mut self, tv1: TypeVar, ty2: T) {
        let tv2 = self.typer.intern(ty2.into());
        self.unify(tv1, tv2);
    }
}

impl Unify<TypeVar, TypeVar> for Context<'_> {
    /// Unifies two type variables `a` and `b`.
    fn union(&mut self, a: TypeVar, b: TypeVar) {
        self.typer.table.union(a, b);
    }
    /// Unifies two polytypes, i.e., types which may contain type variables
    fn unify(&mut self, tv1: TypeVar, tv2: TypeVar) {
        let ty1 = self.typer.lookup(tv1);
        let ty2 = self.typer.lookup(tv2);
        match (&ty1.kind, &ty2.kind) {
            // Unify Unknown types
            (Unknown, Unknown) => self.union(tv1, tv2),
            (Unknown, _) => self.union(tv1, ty2),
            (_, Unknown) => self.union(tv2, ty1),
            // Zip other types and unify their inner Unknown types
            (Nominal(id1), Nominal(id2)) if id1 == id2 => {}
            (Scalar(kind1), Scalar(kind2)) if kind1 == kind2 => {}
            (Optional(tv1), Optional(tv2)) => self.unify(*tv1, *tv2),
            (Stream(tv1), Stream(tv2)) => self.unify(*tv1, *tv2),
            (Set(key1), Set(key2)) => self.unify(*key1, *key2),
            (Vector(elem1), Vector(elem2)) => self.unify(*elem1, *elem2),
            (Array(tv1, sh1), Array(tv2, sh2)) if sh1.dims.len() == sh2.dims.len() => {
                self.unify(*tv1, *tv2);
            }
            (Fun(args1, ret1), Fun(args2, ret2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                    self.unify(*arg1, *arg2);
                }
                self.unify(*ret1, *ret2);
            }
            (Tuple(args1), Tuple(args2)) if args1.len() == args2.len() => {
                for (arg1, arg2) in args1.iter().zip(args2.iter()) {
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
            (Map(key1, val1), Map(key2, val2)) => {
                self.unify(*key1, *key2);
                self.unify(*val1, *val2);
            }
            (Task(params1), Task(params2)) if params1.len() == params2.len() => {
                for (param1, param2) in params1.iter().zip(params2) {
                    self.unify(*param1, *param2);
                }
            }
            _ => self.errors.push(CompilerError::TypeMismatch {
                lhs: ty1.clone(),
                rhs: ty2.clone(),
                span: self.span,
            }),
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
        self.ast.for_each_expr_postorder(|expr| {
            let ctx = &mut Context::new(typer, expr.span, errors, table);
            expr.constrain(ctx)
        });
        self.ast.for_each_fun(|fun| {
            let ctx = &mut Context::new(typer, fun.body.span, errors, table);
            fun.constrain(ctx);
        });
    }
}

trait Constrain<'i> {
    fn constrain(&mut self, context: &mut Context<'i>);
}

impl Constrain<'_> for FunDef {
    /// Constrains the types of a function based on its signature and body.
    fn constrain(&mut self, ctx: &mut Context<'_>) {
        let tv = ctx.table.get_decl(&self.id).tv;
        let ptys = self
            .params
            .iter()
            .map(|id| ctx.table.get_decl(id).tv)
            .collect();
        let rty = self.body.tv;
        let fty = Fun(ptys, rty);
        ctx.unify(tv, fty);
    }
}

// impl Constrain<'_> for (&Ident, &mut TaskDef) {
//     fn constrain(&mut self, ctx: &mut Context<'_>) {
//         let (id, taskdef) = self;
//         for (id, fundef) in taskdef.fundefs.iter() {
//             taskdef.
//         }
//     }
// }

impl Constrain<'_> for Expr {
    /// Constrains an expression based on its subexpressions.
    fn constrain(&mut self, ctx: &mut Context<'_>) {
        match &self.kind {
            Let(id, e1, e2) => {
                let tv = ctx.table.get_decl(id).tv;
                ctx.unify(tv, e1.tv);
                ctx.unify(self.tv, e2.tv);
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
                ctx.unify(self.tv, kind);
            }
            ConsArray(args) => {
                let elem_tv = ctx.typer.fresh();
                let dim = Dim::from(DimVal(args.len() as i32));
                args.iter().for_each(|e| ctx.unify(elem_tv, e.tv));
                let shape = Shape::from(vec![dim]);
                ctx.unify(self.tv, Array(elem_tv, shape));
            }
            ConsStruct(fields) => {
                let fields = fields
                    .iter()
                    .map(|(field, arg)| (*field, arg.tv))
                    .collect::<VecMap<_, _>>();
                ctx.unify(self.tv, Struct(fields));
            }
            ConsVariant(sym, arg) => {
                ctx.unify(self.tv, Enum(vec![(*sym, arg.tv)].into_iter().collect()))
            }
            ConsTuple(args) => {
                let tvs = args.iter().map(|arg| arg.tv).collect();
                ctx.unify(self.tv, Tuple(tvs));
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
                            I8 | I16 | I32 | I64 => ctx.unify(r.tv, I32),
                            F32 => ctx.unify(r.tv, F32),
                            F64 => ctx.unify(r.tv, F64),
                            _ => {}
                        }
                    }
                    if let Scalar(kind) = ctx.typer.lookup(r.tv).kind {
                        match kind {
                            F32 => ctx.unify(l.tv, F32),
                            F64 => ctx.unify(l.tv, F64),
                            _ => {}
                        }
                    }
                }
                Equ | Neq | Gt | Lt | Geq | Leq => {
                    ctx.unify(l.tv, r.tv);
                    ctx.unify(self.tv, Bool)
                }
                Or | And => {
                    ctx.unify(self.tv, l.tv);
                    ctx.unify(self.tv, r.tv);
                    ctx.unify(self.tv, Bool)
                }
                Pipe => todo!(),
                Seq => ctx.unify(self.tv, r.tv),
                BinOpErr => {}
            },
            UnOp(kind, e) => match kind {
                Not => {
                    ctx.unify(self.tv, e.tv);
                    ctx.unify(e.tv, Bool);
                }
                Neg => ctx.unify(self.tv, e.tv),
                Cast(tv) => ctx.unify(e.tv, *tv),
                Project(idx) => {
                    if let Tuple(tvs) = ctx.typer.lookup(e.tv).kind {
                        if let Some(tv) = tvs.get(idx.0) {
                            ctx.unify(self.tv, *tv);
                        } else {
                            ctx.errors
                                .push(CompilerError::OutOfBoundsProject { span: self.span })
                        }
                    }
                }
                Access(sym) => {
                    if let Struct(fields) = ctx.typer.lookup(e.tv).kind {
                        if let Some(tv) = fields.get(sym) {
                            ctx.unify(self.tv, *tv);
                        } else {
                            ctx.errors
                                .push(CompilerError::FieldNotFound { span: self.span })
                        }
                    }
                }
                Call(args) => {
                    // TODO: Figure out how to infer other callables (variants and tasks).
                    // * Variants are not values, so should be possible to figure
                    //   them out during name resolution
                    // * Tasks are values which makes things harder
                    //   * Either do some sophisticated type inference
                    //   * Or maybe represent them as functions
                    let param_tvs = args.iter().map(|arg| arg.tv).collect();
                    ctx.unify(e.tv, Fun(param_tvs, self.tv));
                }
                Emit => {
                    ctx.unify(self.tv, Unit);
                }
                UnOpErr => {}
            },
            If(c, t, e) => {
                ctx.unify(c.tv, Bool);
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
            Loop(_, _) => todo!(),
            ExprErr => {}
        }
    }
}
