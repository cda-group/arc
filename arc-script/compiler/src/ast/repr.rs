//! Internal representation of the `AST`. In comparison to the `HIR`, the `AST` does not have
//! resolved names. Also, all expressions are interned (bump-allocated).

use crate::info::files::ByteIndex;
use crate::info::files::FileId;
use crate::info::files::Loc;

pub(crate) use crate::info::files::Spanned;
pub(crate) use crate::info::names::NameId;
pub(crate) use crate::info::paths::PathId;

use arc_script_compiler_macros::GetId;
use arc_script_compiler_macros::Loc;
use arc_script_compiler_shared::Arena;
use arc_script_compiler_shared::Educe;
use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Into;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::OrdMap;
use arc_script_compiler_shared::Shrinkwrap;

use std::fmt::Debug;
use time::Duration;
use time::PrimitiveDateTime as DateTime;

pub type ExprInterner = Arena<ExprId, ExprKind>;
pub type TypeInterner = Arena<TypeId, TypeKind>;
pub type PatInterner = Arena<PatId, PatKind>;

/// An Arc-AST.
#[derive(Debug, Default)]
pub struct AST {
    pub modules: OrdMap<PathId, Module>,
    pub exprs: ExprInterner,
    pub pats: PatInterner,
    pub types: TypeInterner,
}

/// A module which corresponds directly to a source file.
#[derive(Debug, New)]
pub struct Module {
    pub items: Vec<Item>,
}

/// A set of attributes.
#[derive(Debug, Clone, Loc)]
pub struct Meta {
    pub attrs: Vec<Attr>,
    pub loc: Loc,
}

/// An attribute.
#[derive(Debug, Clone, Loc)]
pub struct Attr {
    pub kind: AttrKind,
    pub loc: Loc,
}

/// A kind of attribute for an item.
#[derive(Debug, Clone)]
pub enum AttrKind {
    Name(Name),
    NameValue(Name, LitKind),
}

pub use crate::info::names::Name;
pub use crate::info::paths::Path;

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
    pub meta: Option<Meta>,
    pub kind: ItemKind,
    pub loc: Loc,
}

/// A kind of item declaration.
#[derive(Debug)]
pub enum ItemKind {
    Assign(Assign),
    Enum(Enum),
    ExternFun(ExternFun),
    ExternType(ExternType),
    Fun(Fun),
    Task(Task),
    TypeAlias(TypeAlias),
    Use(Use),
    Err,
}

/// An item declaration residing within a task.
#[derive(Debug, Loc)]
pub struct ExternTypeItem {
    pub meta: Option<Meta>,
    pub kind: ExternTypeItemKind,
    pub loc: Loc,
}

/// An kind of item declaration residing within an extern type.
#[derive(Debug)]
pub enum ExternTypeItemKind {
    FunDecl(FunDecl),
    Err,
}

/// An item declaration residing within a task.
#[derive(Debug, Loc)]
pub struct TaskItem {
    pub meta: Option<Meta>,
    pub kind: TaskItemKind,
    pub loc: Loc,
}

/// An kind of item declaration residing within a task.
#[derive(Debug)]
pub enum TaskItemKind {
    Enum(Enum),
    ExternFun(ExternFun),
    Fun(Fun),
    Stmt(Stmt),
    TypeAlias(TypeAlias),
    Use(Use),
    Err,
}

/// A task.
#[derive(Debug, New)]
pub struct Task {
    pub name: Name,
    pub params: Vec<Param>,
    pub iinterface: Interface,
    pub ointerface: Interface,
    pub items: Vec<TaskItem>,
}

/// An interface of a task.
#[derive(Debug, New, Loc)]
pub struct Interface {
    pub kind: InterfaceKind,
    pub loc: Loc,
}

/// A kind of an interface.
#[derive(Debug)]
pub enum InterfaceKind {
    Tagged(Vec<Port>),
    // NB: This is lowered into a tagged version.
    Single(Type),
}

/// A local variable or value.
#[derive(Debug, New)]
pub struct Assign {
    pub kind: MutKind,
    pub param: Param,
    pub expr: Expr,
}

#[derive(Debug, Clone, Copy)]
pub enum MutKind {
    Mutable,
    Immutable,
}

/// A function.
#[derive(Debug, New)]
pub struct Fun {
    pub name: Name,
    pub params: Vec<Param>,
    pub rt: Option<Type>,
    pub block: Block,
}

/// A function declaration.
#[derive(Debug, New)]
pub struct FunDecl {
    pub name: Name,
    pub params: Vec<Param>,
    pub rt: Type,
}

/// An externally defined function.
#[derive(Debug, New)]
pub struct ExternFun {
    pub decl: FunDecl,
}

/// An externally defined type.
#[derive(Debug, New)]
pub struct ExternType {
    pub name: Name,
    pub params: Vec<Param>,
    pub items: Vec<ExternTypeItem>,
}

/// A type alias.
#[derive(Debug, New)]
pub struct TypeAlias {
    pub name: Name,
    pub t: Type,
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

/// A block of statements terminated by an expression.
#[derive(Debug, Loc)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Expr>,
    pub loc: Loc,
}

/// A statement.
#[derive(Debug, Loc)]
pub struct Stmt {
    pub kind: StmtKind,
    pub loc: Loc,
}

/// A kind of statement.
#[derive(Debug)]
pub enum StmtKind {
    Empty,
    Assign(Assign),
    Expr(Expr),
}

/// Id of an interned expression.
#[derive(Debug, Copy, Clone, Into, From)]
pub struct ExprId(usize);

/// An expression.
#[derive(Debug, Copy, Clone, Loc, GetId)]
pub struct Expr {
    pub id: ExprId,
    pub loc: Loc,
}

/// A kind of expression.
#[derive(Debug)]
pub enum ExprKind {
    Access(Expr, Name),
    Invoke(Expr, Name, Vec<Expr>),
    After(Expr, Block),
    Every(Expr, Block),
    Array(Vec<Expr>),
    BinOp(Expr, BinOp, Expr),
    Call(Expr, Vec<Expr>),
    Cast(Expr, Type),
    Emit(Expr),
    Enwrap(Path, Expr),
    If(Expr, Block, Option<Block>),
    Is(Path, Expr),
    Lit(LitKind),
    Log(Expr),
    Loop(Block),
    On(Vec<Case>),
    Project(Expr, Index),
    Select(Expr, Vec<Expr>),
    Struct(Vec<Field<Expr>>),
    Tuple(Vec<Expr>),
    UnOp(UnOp, Expr),
    Unwrap(Path, Expr),
    Return(Option<Expr>),
    Break(Option<Expr>),
    Continue,
    Err,
    // NB: These expressions are desugared
    Block(Block),
    Lambda(Vec<Param>, Expr),
    IfAssign(Assign, Block, Option<Block>),
    For(Pat, Expr, Block),
    Match(Expr, Vec<Case>),
    Path(Path, Option<Vec<Type>>),
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
    DateTime(DateTime),
    Duration(Duration),
    Unit,
    Err,
}

/// The arm of a match expression.
#[derive(Debug, Clone, Copy, Loc)]
pub struct Case {
    pub pat: Pat,
    pub body: Expr,
    pub loc: Loc,
}

#[derive(Debug, Copy, Clone, Into, From)]
pub struct PatId(usize);

/// A pattern.
#[derive(Debug, Copy, Clone, Loc, GetId)]
pub struct Pat {
    pub id: PatId,
    pub loc: Loc,
}

/// A kind of pattern.
#[derive(Debug)]
pub enum PatKind {
    Ignore,
    Or(Pat, Pat),
    Struct(Vec<Field<Option<Pat>>>),
    Tuple(Vec<Pat>),
    Const(LitKind),
    Var(Name),
    Variant(Path, Pat),
    By(Pat, Pat),
    Err,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, New, Loc)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub loc: Loc,
}

/// A kind of binary operator.
#[derive(Debug, Clone, Copy)]
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
    Mut,
    Neq,
    Or,
    Pow,
    Sub,
    Xor,
    In,
    RExc,
    RInc,
    Err,
    // NB: These ops are desugared
    By,
    NotIn,
    Pipe,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, Loc)]
pub struct UnOp {
    pub kind: UnOpKind,
    pub loc: Loc,
}

/// A kind of unary operator.
#[derive(Debug, Clone, Copy)]
pub enum UnOpKind {
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
    pub t: Option<Type>,
    pub loc: Loc,
}

/// A port of an interface.
#[derive(Debug, Loc)]
pub struct Port {
    pub name: Name,
    pub t: Type,
    pub loc: Loc,
}

/// A parameter of a function, let-expression, etc.
#[derive(Debug, Loc)]
pub struct Param {
    pub pat: Pat,
    pub t: Option<Type>,
    pub loc: Loc,
}

#[derive(Debug, Copy, Clone, Into, From)]
pub struct TypeId(usize);

/// A type from a type annotation.
#[derive(Debug, Copy, Clone, Loc, GetId)]
pub struct Type {
    pub id: TypeId,
    pub loc: Loc,
}

/// A kind of type from a type annotation.
#[derive(Debug)]
pub enum TypeKind {
    Array(Option<Type>, Shape),
    Fun(Vec<Type>, Type),
    Scalar(ScalarKind),
    Stream(Type),
    Struct(Vec<Field<Type>>),
    Tuple(Vec<Type>),
    Err,
    // NB: These types are lowered
    Path(Path, Option<Vec<Type>>),
    By(Type, Type),
}

/// A kind of scalar type.
#[derive(Debug, Clone, Eq, PartialEq, Copy)]
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
    Str,
    // Eliminated when lowering to HIR
    Unit,
    DateTime,
    Duration,
    Size,
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
