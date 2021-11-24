use crate::ast;
use crate::info::files::Loc;
use crate::prelude::ast::Spanned;

use arc_script_compiler_macros::GetId;
use arc_script_compiler_macros::Loc;
use arc_script_compiler_shared::Arena;
use arc_script_compiler_shared::Educe;
use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Into;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::OrdMap;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecMap;

use bitmaps::Bitmap;
use strum::EnumCount;

use std::collections::VecDeque;

pub(crate) use crate::info::names::NameId;
pub(crate) use crate::info::paths::PathId;

pub(crate) type ExprInterner = Arena<ExprId, ExprKind>;

/// The HIR is a Higher-Level Intermediate Representation more suitable for
/// analysis and code generation than the AST. The main differences between
/// the HIR and AST is that the HIR initially has:
/// * References to names and paths resolved.
/// * Patterns compiled into decision trees.
/// * Definitions stored in maps quick access.
/// Then, after type inference:
/// * Types inferred and definitions monomorphised.
/// After transformations:
/// * Closures eliminated through lambda lifting.
/// * Expressions in SSA form.
#[derive(Debug, Default)]
pub(crate) struct HIR {
    /// Top-level items
    pub(crate) namespace: Vec<PathId>,
    /// Definitions of items.
    pub(crate) defs: OrdMap<PathId, Item>,
    pub(crate) exprs: ExprInterner,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct Item {
    pub(crate) kind: ItemKind,
    pub(crate) loc: Loc,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, From)]
pub(crate) enum ItemKind {
    TypeAlias(TypeAlias),
    Enum(Enum),
    Fun(Fun),
    Task(Task),
    ExternFun(ExternFun),
    ExternType(ExternType),
    Variant(Variant),
}

pub(crate) use ast::Index;
pub(crate) use ast::Name;
pub(crate) use ast::Path;

#[derive(Debug, Clone, New)]
pub(crate) struct Fun {
    pub(crate) path: Path,
    pub(crate) kind: FunKind,
    pub(crate) params: Vec<Param>,
    pub(crate) body: Block,
    pub(crate) t: Type,
    pub(crate) rt: Type,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FunKind {
    Free,   // Freestanding function
    Method, // Method function
}

#[derive(Debug, Clone, New)]
pub(crate) struct ExternFun {
    pub(crate) path: Path,
    pub(crate) kind: FunKind,
    pub(crate) params: Vec<Param>,
    pub(crate) t: Type,
    pub(crate) rt: Type,
}

#[derive(Debug, Clone, New)]
pub(crate) struct ExternType {
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) items: Vec<PathId>,
    pub(crate) t: Type
}

#[derive(Debug, Clone, Copy, New, Loc)]
pub(crate) struct Param {
    pub(crate) kind: ParamKind,
    pub(crate) t: Type,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ParamKind {
    Ok(Name),
    Ignore,
    Err,
}

#[derive(Debug, Clone, New)]
pub(crate) struct Enum {
    pub(crate) path: Path,
    pub(crate) variants: Vec<Path>,
}

#[derive(Debug, Clone)]
pub(crate) enum Vis {
    Private,
    Public,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct Variant {
    pub(crate) path: Path,
    pub(crate) t: Type,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, New)]
pub(crate) struct TypeAlias {
    pub(crate) path: Path,
    pub(crate) t: Type,
}

#[derive(Debug, Clone, New)]
pub(crate) struct Assign {
    pub(crate) kind: MutKind,
    pub(crate) param: Param,
    pub(crate) expr: Expr,
}

pub(crate) use ast::MutKind;

#[derive(Debug, Clone, New)]
pub(crate) struct Task {
    pub(crate) path: Path,
    /// Task-constructor function
    pub(crate) cons_x: Path,
    /// Type of the task-constructor: Params -> Streams -> Streams
    pub(crate) cons_t: Type,
    /// Type of the task-function: Streams -> Streams
    pub(crate) fun_t: Type,
    /// Type of the task-struct: {Params, Assignments}
    pub(crate) struct_t: Type,
    /// Initializer parameters of the task.
    pub(crate) params: Vec<Param>,
    /// Assigned variables of the task.
    pub(crate) fields: VecMap<Name, Type>,
    /// Input interface to the task.
    pub(crate) iinterface: Interface,
    /// Output interface to the task.
    pub(crate) ointerface: Interface,
    /// Event handler.
    pub(crate) on_event: OnEvent,
    /// Statements run at startup.
    pub(crate) on_start: OnStart,
    /// Items of the task.
    pub(crate) namespace: Vec<PathId>,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct Interface {
    pub(crate) interior: Path,
    pub(crate) exterior: Vec<Type>,
    pub(crate) keys: Vec<Type>,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct OnEvent {
    pub(crate) fun: Path, // Event handler function
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct OnStart {
    pub(crate) fun: Path, // Startup function
    pub(crate) loc: Loc,
}

pub(crate) use ast::Attr;
pub(crate) use ast::AttrKind;
pub(crate) use ast::Meta;

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct Block {
    /// This is a VecDeque so we can prepend statements.
    pub(crate) stmts: VecDeque<Stmt>,
    pub(crate) var: Var,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, New, Loc)]
pub(crate) struct Stmt {
    pub(crate) kind: StmtKind,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone)]
pub(crate) enum StmtKind {
    Assign(Assign),
}

#[derive(Debug, Clone, Copy, New, Loc)]
pub(crate) struct Var {
    pub(crate) kind: VarKind,
    pub(crate) t: Type,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum VarKind {
    Ok(Name, ScopeKind),
    Err,
}

/// Id of an interned expression.
#[derive(Debug, Copy, Clone, Into, From)]
pub struct ExprId(usize);

#[derive(Debug, Clone, Copy, New, Loc, GetId)]
pub(crate) struct Expr {
    pub(crate) id: ExprId,
    pub(crate) t: Type,
    pub(crate) loc: Loc,
}

#[derive(Debug, Clone)]
pub(crate) enum ExprKind {
    Access(Var, Name),
    After(Var, Block),
    Array(Vec<Var>),
    BinOp(Var, BinOp, Var),
    Call(Var, Vec<Var>),
    SelfCall(Path, Vec<Var>),
    Invoke(Var, Name, Vec<Var>),
    Cast(Var, Type),
    Emit(Var),
    Enwrap(Path, Var), // Construct a variant
    Every(Var, Block),
    If(Var, Block, Block),
    Is(Path, Var), // Check a variant
    Lit(LitKind),
    Log(Var),
    Loop(Block),
    Project(Var, Index),
    Select(Var, Vec<Var>),
    Struct(VecMap<Name, Var>),
    Tuple(Vec<Var>),
    UnOp(UnOp, Var),
    Unwrap(Path, Var), // Deconstruct a variant
    Return(Var),
    Break(Var),
    Continue,
    Err,
    // Expressions constructed by lowering
    Item(Path),            // Item reference
    Unreachable,           // Unreachable code
    Initialise(Name, Var), // Initialise state-variable
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ScopeKind {
    Local,  // A variable which is bound inside a function-scope
    Member, // A variable which is bound inside a task-scope
    Global, // A variable which is bound inside the global-scope
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
    Err,
}

pub(crate) use ast::UnOp;
pub(crate) use ast::UnOpKind;

#[derive(Shrinkwrap, New, From, Default, Debug, Eq, PartialEq, Copy, Clone, Hash, GetId)]
pub struct Type {
    pub(crate) id: TypeId,
}

/// A type variable which maps to a [`crate::repr::hir::Type`].
#[derive(Shrinkwrap, Default, Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub struct TypeId(pub(crate) u32);

#[derive(Debug, Clone)]
pub enum TypeKind {
    Array(Type, Shape),
    Fun(Vec<Type>, Type),
    Nominal(Path),
    Scalar(ScalarKind),
    Stream(Type),
    Struct(VecMap<Name, Type>),
    Tuple(Vec<Type>),
    Unknown(Constraint),
    Err,
}

pub(crate) use ast::ScalarKind;

/// A constraint on types imposed by various operations.
#[derive(Debug, Default, Clone, Eq, PartialEq, Shrinkwrap, From)]
#[shrinkwrap(mutable)]
pub struct Constraint(pub Bitmap<{ ConstraintKind::COUNT }>);

/// Type constraints imposed by various operations.
#[derive(Debug, Clone, Eq, PartialEq, EnumCount)]
#[repr(usize)]
pub(crate) enum ConstraintKind {
    Addable,    // i*, u*, f*, str
    Negatable,  // i*, u*, f*, duration
    Numeric,    // i*, u*, f*
    Comparable, // i*, u*, f*, str, duration, time
    Equatable,  // i*, u*, f*, str, duration, time, bytes, array, record, bool
    Record,     // record
    Timeable,   // duration, time
    Stringable, // i*, u*, f*, str, time, duration, bool
}

#[derive(Debug, Clone, New)]
pub struct Shape {
    pub(crate) dims: Vec<Dim>,
}

#[derive(Debug, Clone, New)]
pub struct Dim {
    pub(crate) kind: DimKind,
}

#[derive(Debug, Clone)]
pub enum DimKind {
    Op(Box<Dim>, DimOp, Box<Dim>),
    Val(i32),
    Var(i32),
    Err,
}

#[derive(Debug, Clone)]
pub struct DimOp {
    pub(crate) kind: DimOpKind,
}

pub(crate) use ast::DimOpKind;

pub(crate) use ast::LitKind;
