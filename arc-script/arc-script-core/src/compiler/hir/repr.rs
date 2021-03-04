use crate::compiler::ast;

use crate::compiler::info::files::Loc;

use arc_script_core_macros::Spanned;
use arc_script_core_shared::Educe;
use arc_script_core_shared::New;
use arc_script_core_shared::OrdMap;
use arc_script_core_shared::VecMap;

use crate::prelude::ast::Spanned;

pub(crate) use crate::compiler::info::names::NameId;
pub(crate) use crate::compiler::info::paths::PathId;
pub(crate) use crate::compiler::info::types::TypeId;

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
    pub(crate) items: Vec<Path>,
    /// Definitions of items.
    pub(crate) defs: OrdMap<Path, Item>,
}

#[derive(New, Spanned, Debug)]
pub(crate) struct Item {
    pub(crate) kind: ItemKind,
    pub(crate) loc: Option<Loc>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum ItemKind {
    Alias(Alias),
    Enum(Enum),
    Fun(Fun),
    State(State),
    Task(Task),
    Extern(Extern),
    Variant(Variant),
}

pub(crate) use ast::Name;

pub(crate) use ast::Index;

/// A path of names.
#[derive(Debug, Copy, Clone, Educe, New, Spanned)]
#[educe(PartialEq, Eq, Hash)]
pub struct Path {
    pub(crate) id: PathId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    pub(crate) loc: Option<Loc>,
}

#[derive(New, Debug)]
pub(crate) struct Fun {
    pub(crate) kind: FunKind,
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) channels: Option<Vec<Param>>,
    pub(crate) body: Expr,
    pub(crate) tv: TypeId,
    pub(crate) rtv: TypeId,
}

#[derive(Debug)]
pub(crate) enum FunKind {
    Global,
    Method,
}

#[derive(New, Debug)]
pub(crate) struct Extern {
    pub(crate) kind: FunKind,
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) tv: TypeId,
    pub(crate) rtv: TypeId,
}

#[derive(New, Spanned, Debug, Clone, Copy)]
pub(crate) struct Param {
    pub(crate) kind: ParamKind,
    pub(crate) tv: TypeId,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ParamKind {
    Var(Name),
    Ignore,
    Err,
}

#[derive(New, Debug)]
pub(crate) struct Enum {
    pub(crate) path: Path,
    pub(crate) variants: Vec<Path>,
}

#[derive(New, Spanned, Debug)]
pub(crate) struct Variant {
    pub(crate) path: Path,
    pub(crate) tv: TypeId,
    pub(crate) loc: Option<Loc>,
}

#[derive(New, Debug)]
pub(crate) struct Alias {
    pub(crate) path: Path,
    pub(crate) tv: TypeId,
}

#[derive(New, Debug)]
pub(crate) struct State {
    pub(crate) path: Path,
    pub(crate) tv: TypeId,
    pub(crate) init: Expr,
}

/// A task is a generic low-level primitive which resembles a node in the dataflow graph.
#[derive(New, Debug)]
pub(crate) struct Task {
    pub(crate) path: Path,
    /// Type of the task.
    pub(crate) tv: TypeId,
    /// Side-input parameters to the task.
    pub(crate) params: Vec<Param>,
    /// Constructor which also flattens the input parameters of a task when initializing it.
    ///
    ///   task Foo((a, b): (i32, f32)) (I(i32)) -> (O(i32))
    ///       fun foo() -> i32 { a + b }
    ///       on I(c) => emit O(foo() + c)
    ///   end
    ///
    /// Lowers into:
    ///
    /// fun Foo(x: (i32, i32)) -> Task((i32) -> (i32)) {
    ///     let a = x.0 in
    ///     let b = x.1 in
    ///     _Foo(a, b)
    /// }
    /// task _Foo(a: i32, b: i32) (I(i32)) -> (O(i32)) {
    ///     fun foo() { a + b }
    ///     on I(c) => emit Out(foo() + c)
    /// }
    /// Input hub to the task.
    pub(crate) ihub: Hub,
    /// Input hub to the task.
    pub(crate) ohub: Hub,
    /// Event handler.
    pub(crate) on: Option<On>,
    /// Items of the task.
    pub(crate) items: Vec<Path>,
}

#[derive(Debug, New, Spanned)]
pub(crate) struct Hub {
    pub(crate) tv: TypeId,
    pub(crate) kind: HubKind,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug)]
pub(crate) enum HubKind {
    Tagged(Path),
    Single(TypeId),
}

#[derive(New, Spanned, Debug)]
pub(crate) struct On {
    pub(crate) param: Param,
    pub(crate) body: Expr,
    pub(crate) loc: Option<Loc>,
}

#[derive(New, Spanned, Debug)]
pub(crate) struct Setting {
    pub(crate) kind: SettingKind,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug)]
pub(crate) enum SettingKind {
    Activate(ast::Name),
    Calibrate(ast::Name, LitKind),
}

#[derive(Debug, New, Clone, Spanned)]
pub(crate) struct Expr {
    pub(crate) kind: ExprKind,
    pub(crate) tv: TypeId,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug, Clone)]
pub(crate) enum ExprKind {
    Access(Box<Expr>, Name),
    Array(Vec<Expr>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    Break,
    Call(Box<Expr>, Vec<Expr>),
    Emit(Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Item(Path),
    Let(Param, Box<Expr>, Box<Expr>),
    Lit(LitKind),
    Log(Box<Expr>),
    Loop(Box<Expr>),
    Project(Box<Expr>, Index),
    Struct(VecMap<Name, Expr>),
    Tuple(Vec<Expr>),
    UnOp(UnOp, Box<Expr>),
    Var(Name, VarKind),
    Enwrap(Path, Box<Expr>), // Construct a variant
    Unwrap(Path, Box<Expr>), // Deconstruct a variant
    Is(Path, Box<Expr>),     // Check a variant
    Return(Box<Expr>),
    Todo,
    Err,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum VarKind {
    Local,
    Member,
    // Global,
}

#[derive(Debug, New, Clone, Spanned)]
pub(crate) struct BinOp {
    pub(crate) kind: BinOpKind,
    pub(crate) loc: Option<Loc>,
}

pub(crate) use ast::BinOpKind;

#[derive(Debug, New, Clone, Spanned)]
pub(crate) struct UnOp {
    pub(crate) kind: UnOpKind,
    pub(crate) loc: Option<Loc>,
}

pub(crate) use ast::UnOpKind;

#[derive(Debug, Clone, New)]
pub struct Type {
    pub(crate) kind: TypeKind,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Array(TypeId, Shape),
    Fun(Vec<TypeId>, TypeId),
    Map(TypeId, TypeId),
    Nominal(Path),
    Optional(TypeId),
    Scalar(ScalarKind),
    Set(TypeId),
    Stream(TypeId),
    Struct(VecMap<Name, TypeId>),
    Tuple(Vec<TypeId>),
    Unknown,
    Vector(TypeId),
    Boxed(TypeId),
    By(TypeId, TypeId),
    Err,
}

pub(crate) use ast::ScalarKind;

#[derive(Debug, Clone, New)]
pub struct Shape {
    pub(crate) dims: Vec<Dim>,
}

#[derive(Debug, Clone, New)]
pub struct Dim {
    pub(crate) kind: DimKind,
}

/// An expression solveable by the z3 SMT solver
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
