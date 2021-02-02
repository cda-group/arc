use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::files::{ByteIndex, FileId, Loc, Span};
use crate::compiler::info::names::NameId;
use crate::compiler::info::modes::Mode;
use crate::compiler::info::paths::PathBuf;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;
use crate::compiler::shared::{Map, New};

// use bumpalo::collections::Vec;
// use bumpalo::boxed::Box;
use arc_script_macros::Spanned;
use time::Duration;
use std::fmt::Debug;

/// A structure which keeps the start and end position of an AST node plus its source file.
/// NB: The FileId could probably be kept in a nicer place to take up less space.
#[derive(New)]
pub struct Spanned<Node>(pub FileId, pub ByteIndex, pub Node, pub ByteIndex);

/// An Arc-AST.
#[derive(Debug, Default)]
pub struct AST {
    pub modules: Map<PathId, Module>,
    pub exprs: ExprInterner,
}

#[derive(Debug, Default)]
pub struct ModuleInterner {
    pub(crate) store: Map<PathId, Module>,
}

/// A module which corresponds directly to a source file.
#[derive(Debug, New)]
pub struct Module {
    pub items: Vec<Item>,
}

/// A setting for an item.
#[derive(Debug, Spanned)]
pub struct Setting {
    pub kind: SettingKind,
    pub loc: Option<Loc>,
}

/// A kind of setting for an item.
#[derive(Debug)]
pub enum SettingKind {
    Activate(Name),
    Calibrate(Name, LitKind),
}

#[derive(Debug, Spanned, Clone)]
pub struct Path {
    pub id: PathId,
    pub loc: Option<Loc>,
}

/// An identifier.
#[derive(Debug, Clone, Copy, Spanned, Educe, New)]
#[educe(PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name {
    pub id: NameId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub loc: Option<Loc>,
}

/// An index into a tuple.
#[derive(Debug, Copy, Clone, Educe, New, Spanned)]
#[educe(PartialEq, Eq, Hash)]
pub struct Index {
    pub id: usize,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    pub loc: Option<Loc>,
}

/// An item declaration.
#[derive(Debug, Spanned)]
pub struct Item {
    pub kind: ItemKind,
    pub settings: Option<Vec<Setting>>,
    pub loc: Option<Loc>,
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
    On(On),
    State(State),
    Use(Use),
    Err,
}

/// A task.
#[derive(Debug, New)]
pub struct Task {
    pub name: Name,
    pub params: Vec<Param>,
    pub iports: Vec<Variant>,
    pub oports: Vec<Variant>,
    pub items: Vec<TaskItem>,
}

/// A function.
#[derive(Debug, New)]
pub struct Fun {
    pub name: Name,
    pub params: Vec<Param>,
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

/// A state variable.
#[derive(Debug, New)]
pub struct State {
    pub name: Name,
    pub expr: Expr,
}

/// An item declaration residing within a task.
#[derive(Debug, Spanned)]
pub struct TaskItem {
    pub kind: TaskItemKind,
    pub settings: Option<Vec<Setting>>,
    pub loc: Option<Loc>,
}

#[derive(Debug, Default)]
pub struct ExprInterner {
    pub(crate) store: Vec<ExprKind>,
}

#[derive(Debug, Copy, Clone)]
pub struct ExprId(usize);

/// An expression.
#[derive(Debug, Spanned)]
pub struct Expr {
    pub id: ExprId,
    pub loc: Option<Loc>,
}

impl ExprInterner {
    pub fn intern(&mut self, expr: ExprKind) -> ExprId {
        let id = ExprId(self.store.len());
        self.store.push(expr);
        id
    }

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
    Cast(Expr, Type),
    Emit(Expr),
    For(Pat, Expr, Expr),
    If(Expr, Expr, Expr),
    IfLet(Pat, Expr, Expr, Expr),
    Lambda(Vec<Param>, Expr),
    Let(Param, Expr, Expr),
    Lit(LitKind),
    Log(Expr),
    Loop(Expr),
    Reduce(Pat, Expr, ReduceKind),
    Unwrap(Name, Expr),
    Is(Name, Expr),
    Enwrap(Path, Expr),
    Match(Expr, Vec<Case>),
    Path(Path),
    Project(Expr, Index),
    Struct(Vec<Field<Expr>>),
    Tuple(Vec<Expr>),
    UnOp(UnOp, Expr),
    Return(Option<Expr>),
    Err,
}

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
    Time(Duration),
    Unit,
    Err,
}

/// The arm of a match expression.
#[derive(Debug, Spanned)]
pub struct Case {
    pub pat: Pat,
    pub body: Expr,
    pub loc: Option<Loc>,
}

/// A pattern.
#[derive(Debug, Spanned)]
pub struct Pat {
    pub kind: PatKind,
    pub loc: Option<Loc>,
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
    Variant(Name, Box<Pat>),
    Err,
}

/// A binary operator.
#[derive(Debug, Spanned)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub loc: Option<Loc>,
}

/// A kind of binary operator.
#[derive(Debug, Clone)]
pub enum BinOpKind {
    Add,
    And,
    Band,
    Bor,
    Bxor,
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
    Pow,
    Seq,
    Sub,
    Xor,
    Err,
}

/// A unary operator.
#[derive(Debug, Spanned)]
pub struct UnOp {
    pub kind: UnOpKind,
    pub loc: Option<Loc>,
}

/// A kind of unary operator.
#[derive(Debug, Clone)]
pub enum UnOpKind {
    Neg,
    Not,
    Err,
}

/// A field of a struct.
#[derive(Debug, Spanned)]
pub struct Field<T: Debug> {
    pub name: Name,
    pub val: T,
    pub loc: Option<Loc>,
}

/// A variant of an enum or port.
#[derive(Debug, Spanned)]
pub struct Variant {
    pub name: Name,
    pub ty: Option<Type>,
    pub loc: Option<Loc>,
}

/// A parameter of a function, let-expression, etc.
#[derive(Debug, Spanned)]
pub struct Param {
    pub pat: Pat,
    pub ty: Option<Type>,
    pub loc: Option<Loc>,
}

/// A type from a type annotation.
#[derive(Debug, Spanned)]
pub struct Type {
    pub kind: TypeKind,
    pub loc: Option<Loc>,
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
    Task(Vec<Type>, Vec<Type>),
    Tuple(Vec<Type>),
    Vector(Box<Type>),
    Err,
}

/// A kind of scalar type.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ScalarKind {
    Bool,
    Char,
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
    Bot,
}

/// A shape of an array.
#[derive(Debug, Spanned)]
pub struct Shape {
    pub dims: Vec<Dim>,
    pub loc: Option<Loc>,
}

/// A dimension of a shape.
#[derive(Debug, Spanned)]
pub struct Dim {
    pub kind: DimKind,
    pub loc: Option<Loc>,
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
#[derive(Debug, Spanned)]
pub struct DimOp {
    pub kind: DimOpKind,
    pub loc: Option<Loc>,
}

/// A kind of dimension operator.
#[derive(Debug, Clone)]
pub enum DimOpKind {
    Add,
    Div,
    Mul,
    Sub,
}
