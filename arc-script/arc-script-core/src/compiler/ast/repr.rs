//! Internal representation of the `AST`. In comparison to the `HIR`, the `AST` does not have
//! resolved names. Also, all expressions are interned (bump-allocated).

use crate::compiler::info::files::{ByteIndex, FileId, Loc};
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;

use arc_script_core_macros::Loc;
use arc_script_core_shared::Educe;
use arc_script_core_shared::New;
use arc_script_core_shared::OrdMap;

use half::bf16;
use half::f16;
use std::fmt::Debug;
use time::Duration;
use time::PrimitiveDateTime as DateTime;

/// A structure which keeps the start and end position of an AST node plus its source file.
/// NB: The `FileId` could probably be kept in a nicer place to take up less space.
#[derive(Debug, New)]
pub struct Spanned<Node>(pub FileId, pub ByteIndex, pub Node, pub ByteIndex);

/// An Arc-AST.
#[derive(Debug, Default)]
pub struct AST {
    pub modules: OrdMap<PathId, Module>,
    pub exprs: ExprInterner,
}

/// A module which corresponds directly to a source file.
#[derive(Debug, New)]
pub struct Module {
    pub items: Vec<Item>,
}

/// A setting for an item.
#[derive(Debug, Loc)]
pub struct Setting {
    pub kind: SettingKind,
    pub loc: Loc,
}

/// A kind of setting for an item.
#[derive(Debug)]
pub enum SettingKind {
    Activate(Name),
    Calibrate(Name, LitKind),
}

/// Path to an item or variable.
#[derive(Debug, Loc, Clone, Copy)]
pub struct Path {
    pub id: PathId,
    pub loc: Loc,
}

/// An identifier.
#[derive(Debug, Clone, Copy, Loc, Educe, New)]
#[educe(PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name {
    pub id: NameId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub loc: Loc,
}

/// An index into a tuple.
#[derive(Debug, Copy, Clone, Educe, New, Loc)]
#[educe(PartialEq, Eq, Hash)]
pub struct Index {
    pub id: usize,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    pub loc: Loc,
}

/// An item declaration.
#[derive(Debug, Loc)]
pub struct Item {
    pub kind: ItemKind,
    pub settings: Option<Vec<Setting>>,
    pub loc: Loc,
}

/// A kind of item declaration.
#[derive(Debug)]
pub enum ItemKind {
    Alias(Alias),
    Enum(Enum),
    Fun(Fun),
    Extern(Extern),
    Task(Task),
    Use(Use),
    Err,
}

/// An kind of item declaration residing within a task.
#[derive(Debug)]
pub enum TaskItemKind {
    Alias(Alias),
    Enum(Enum),
    Fun(Fun),
    Extern(Extern),
    On(On),
    State(State),
    Startup(Startup),
    Use(Use),
    Timer(Timer),
    Timeout(Timeout),
    Err,
}

/// A task.
#[derive(Debug, New)]
pub struct Task {
    pub name: Name,
    pub params: Vec<Param>,
    pub ihub: Hub,
    pub ohub: Hub,
    pub items: Vec<TaskItem>,
}

/// A hub of a task.
#[derive(Debug, New, Loc)]
pub struct Hub {
    pub kind: HubKind,
    pub loc: Loc,
}

/// A kind of hub.
#[derive(Debug)]
pub enum HubKind {
    Tagged(Vec<Port>),
    Single(Type),
}

/// A function.
#[derive(Debug, New)]
pub struct Fun {
    pub name: Name,
    pub params: Vec<Param>,
    pub channels: Option<Vec<Param>>,
    pub return_ty: Option<Type>,
    pub body: Expr,
}

/// An extern function.
#[derive(Debug, New)]
pub struct Extern {
    pub name: Name,
    pub params: Vec<Param>,
    pub return_ty: Type,
}

/// A type alias.
#[derive(Debug, New)]
pub struct Alias {
    pub name: Name,
    pub ty: Type,
}

/// An enum.
#[derive(Debug, New)]
pub struct Enum {
    pub name: Name,
    pub variants: Vec<Variant>,
}

/// An import.
#[derive(Debug, New)]
pub struct Use {
    pub path: Path,
    pub alias: Option<Name>,
}

/// An event handler.
#[derive(Debug, New)]
pub struct On {
    pub cases: Vec<Case>,
}

/// A timer declaration.
#[derive(Debug, New)]
pub struct Timer {
    pub ty: Type,
}

/// A trigger handler.
#[derive(Debug, New)]
pub struct Timeout {
    pub cases: Vec<Case>,
}

/// A timer declaration.
#[derive(Debug, New)]
pub struct Startup {
    pub expr: Expr,
}

/// A state variable.
#[derive(Debug, New)]
pub struct State {
    pub param: Param,
    pub expr: Expr,
}

/// An item declaration residing within a task.
#[derive(Debug, Loc)]
pub struct TaskItem {
    pub kind: TaskItemKind,
    pub settings: Option<Vec<Setting>>,
    pub loc: Loc,
}

/// Data structure which interns (bump-allocates) expressions.
#[derive(Debug, Default)]
pub struct ExprInterner {
    pub(crate) store: Vec<ExprKind>,
}

/// Id of an interned expression.
#[derive(Debug, Copy, Clone)]
pub struct ExprId(usize);

/// An expression.
#[derive(Debug, Loc)]
pub struct Expr {
    pub id: ExprId,
    pub loc: Loc,
}

impl ExprInterner {
    /// Interns an expression and returns a reference to it.
    pub fn intern(&mut self, expr: ExprKind) -> ExprId {
        let id = ExprId(self.store.len());
        self.store.push(expr);
        id
    }

    /// Resolves an `ExprId` into its associated `ExprKind`.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn resolve(&self, id: ExprId) -> &ExprKind {
        self.store.get(id.0).unwrap()
    }
}

/// A kind of expression.
#[derive(Debug)]
pub enum ExprKind {
    Access(Expr, Name),
    Array(Vec<Expr>),
    BinOp(Expr, BinOp, Expr),
    Break,
    Call(Expr, Vec<Expr>),
    Select(Expr, Vec<Expr>),
    Empty,
    Cast(Expr, Type),
    Emit(Expr),
    Trigger(Expr),
    For(Pat, Expr, Expr),
    If(Expr, Expr, Expr),
    IfLet(Pat, Expr, Expr, Expr),
    Lambda(Vec<Param>, Expr),
    Let(Param, Expr, Expr),
    Lit(LitKind),
    Log(Expr),
    Loop(Expr),
    Reduce(Pat, Expr, ReduceKind),
    Unwrap(Path, Expr),
    Is(Path, Expr),
    Enwrap(Path, Expr),
    Match(Expr, Vec<Case>),
    Path(Path),
    Project(Expr, Index),
    Struct(Vec<Field<Expr>>),
    Tuple(Vec<Expr>),
    UnOp(UnOp, Expr),
    Return(Option<Expr>),
    Todo,
    Err,
}

/// A kind of reduce-expression.
#[derive(Debug)]
pub enum ReduceKind {
    Loop(Expr),
    For(Pat, Expr, Expr),
}

/// A literal.
#[derive(Debug, Clone)]
pub enum LitKind {
    Bool(bool),
    Char(char),
    Bf16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Str(String),
    DateTime(DateTime),
    Duration(Duration),
    Unit,
    Err,
}

/// The arm of a match expression.
#[derive(Debug, Loc)]
pub struct Case {
    pub pat: Pat,
    pub body: Expr,
    pub loc: Loc,
}

/// A pattern.
#[derive(Debug, Loc)]
pub struct Pat {
    pub kind: PatKind,
    pub loc: Loc,
}

/// A kind of pattern.
#[derive(Debug)]
pub enum PatKind {
    Ignore,
    Or(Box<Pat>, Box<Pat>),
    Struct(Vec<Field<Option<Pat>>>),
    Tuple(Vec<Pat>),
    Val(LitKind),
    Var(Name),
    Variant(Path, Box<Pat>),
    By(Box<Pat>, Box<Pat>),
    After(Box<Pat>, Box<Pat>),
    Err,
}

/// A binary operator.
#[derive(Debug, Loc)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub loc: Loc,
}

/// A kind of binary operator.
#[derive(Debug, Clone)]
pub enum BinOpKind {
    After,
    Add,
    And,
    Band,
    Bor,
    Bxor,
    By,
    In,
    NotIn,
    Div,
    Equ,
    Geq,
    Gt,
    Leq,
    Lt,
    Mod,
    Mul,
    Neq,
    Or,
    Pipe,
    Mut,
    Pow,
    Seq,
    Sub,
    Xor,
    Err,
}

/// A unary operator.
#[derive(Debug, Loc)]
pub struct UnOp {
    pub kind: UnOpKind,
    pub loc: Loc,
}

/// A kind of unary operator.
#[derive(Debug, Clone)]
pub enum UnOpKind {
    Add,
    Boxed,
    Del,
    Neg,
    Not,
    Err,
}

/// A field of a struct.
#[derive(Debug, Loc)]
pub struct Field<T: Debug> {
    pub name: Name,
    pub val: T,
    pub loc: Loc,
}

/// A variant of an enum.
#[derive(Debug, Loc)]
pub struct Variant {
    pub name: Name,
    pub ty: Option<Type>,
    pub loc: Loc,
}

/// A port of a hub.
#[derive(Debug, Loc)]
pub struct Port {
    pub name: Name,
    pub ty: Type,
    pub loc: Loc,
}

/// A parameter of a function, let-expression, etc.
#[derive(Debug, Loc)]
pub struct Param {
    pub pat: Pat,
    pub ty: Option<Type>,
    pub loc: Loc,
}

/// A type from a type annotation.
#[derive(Debug, Loc)]
pub struct Type {
    pub kind: TypeKind,
    pub loc: Loc,
}

/// A kind of type from a type annotation.
#[derive(Debug)]
pub enum TypeKind {
    Array(Option<Box<Type>>, Shape),
    Fun(Vec<Type>, Box<Type>),
    Map(Box<Type>, Box<Type>),
    Nominal(Path),
    Optional(Box<Type>),
    Scalar(ScalarKind),
    Set(Box<Type>),
    Stream(Box<Type>),
    Struct(Vec<Field<Type>>),
    Tuple(Vec<Type>),
    Vector(Box<Type>),
    Boxed(Box<Type>),
    By(Box<Type>, Box<Type>),
    After(Box<Type>, Box<Type>),
    Err,
}

/// A kind of scalar type.
#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum ScalarKind {
    Bool,
    Char,
    Bf16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Null,
    Str,
    Unit,
    Never,
    Bot,
    DateTime,
    Duration,
}

/// A shape of an array.
#[derive(Debug, Loc)]
pub struct Shape {
    pub dims: Vec<Dim>,
    pub loc: Loc,
}

/// A dimension of a shape.
#[derive(Debug, Loc)]
pub struct Dim {
    pub kind: DimKind,
    pub loc: Loc,
}

/// A kind of dimension of a shape.
#[derive(Debug)]
pub enum DimKind {
    Op(Box<Dim>, DimOp, Box<Dim>),
    Val(i32),
    Var(i32),
    Err,
}

/// A dimension operator.
#[derive(Debug, Loc)]
pub struct DimOp {
    pub kind: DimOpKind,
    pub loc: Loc,
}

/// A kind of dimension operator.
#[derive(Debug, Clone)]
pub enum DimOpKind {
    Add,
    Div,
    Mul,
    Sub,
}
