use crate::compiler::ast;
use crate::compiler::hir::{Name, Path, Type};
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::shared::{Map, New, Set, VecMap};

use educe::Educe;
use half::bf16;
use half::f16;
use time::Duration;

/// MLIR is a Multi-Level Intermediate Representation which is the final
/// representation within the Arc-Script compiler. In contrast to the HIR,
/// MLIR is in SSA form.
#[derive(Debug, New)]
pub(crate) struct MLIR {
    /// Top-level items
    pub(crate) items: Vec<Path>,
    /// Definitions of items.
    pub(crate) defs: Map<Path, Item>,
//    /// Main function for generating the dataflow.
//     pub(crate) main: Fun,
}

#[derive(New, Debug)]
pub(crate) struct Item {
    pub(crate) kind: ItemKind,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug)]
pub(crate) enum ItemKind {
    Enum(Enum),
    Fun(Fun),
    State(State),
    Task(Task),
}

#[derive(New, Debug)]
pub(crate) struct Fun {
    pub(crate) name: Name,
    pub(crate) params: Vec<Var>,
    pub(crate) body: Region,
    pub(crate) tv: TypeId,
}

#[derive(New, Debug, Copy, Clone)]
pub(crate) struct Var {
    pub(crate) name: Name,
    pub(crate) tv: TypeId,
}

#[derive(New, Debug)]
pub(crate) struct Enum {
    pub(crate) name: Name,
    pub(crate) variants: Vec<Variant>,
}

#[derive(New, Debug)]
pub(crate) struct Variant {
    pub(crate) name: Name,
    pub(crate) tv: TypeId,
    pub(crate) loc: Option<Loc>,
}

#[derive(New, Debug)]
pub(crate) struct Alias {
    pub(crate) name: Name,
    pub(crate) tv: TypeId,
}

#[derive(New, Debug)]
pub(crate) struct State {
    pub(crate) name: Name,
    pub(crate) tv: TypeId,
    pub(crate) init: Op,
}

/// A task is a generic low-level primitive which resembles a node in the dataflow graph.
#[derive(New, Debug)]
pub(crate) struct Task {
    pub(crate) name: Name,
    /// Type of the task.
    pub(crate) tv: TypeId,
    /// Side-input parameters to the task.
    pub(crate) params: Vec<Var>,
    /// Input ports to the task.
    pub(crate) iports: TypeId,
    /// Output ports of the task.
    pub(crate) oports: TypeId,
    /// Event handler.
    pub(crate) handler: Op,
    /// Items of the task.
    pub(crate) items: Vec<Path>,
}

#[derive(New, Debug)]
pub(crate) struct Setting {
    pub(crate) kind: SettingKind,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug)]
pub(crate) enum SettingKind {
    Activate(ast::Name),
    Calibrate(ast::Name, ConstKind),
}

#[derive(Debug, New)]
pub(crate) struct Op {
    pub(crate) var: Option<Var>,
    pub(crate) kind: OpKind,
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug)]
pub(crate) enum OpKind {
    // Task-ops
    Access(Var, Name),
    Array(Vec<Var>),
    BinOp(TypeId, Var, BinOp, Var),
    Break,
    Call(Path, Vec<Var>),
    CallIndirect(Var, Vec<Var>),
    Const(ConstKind),
    Emit(Var),
    If(Var, Region, Region),
    Log(Var),
    Loop(Var),
    Project(Var, usize),
    Res(Var),
    Return(Var),
    Struct(VecMap<Name, Var>),
    Tuple(Vec<Var>),
    UnOp(UnOp, Var),
    Enwrap(Path, Var),
    Unwrap(Path, Var),
    Is(Path, Var),
    // Dataflow-ops
    Edge((Var, usize), (Var, usize)),
    Node(Path, Vec<Var>),
}

#[derive(Debug, New)]
pub(crate) struct Region {
    pub(crate) blocks: Vec<Block>,
}

#[derive(Debug, New)]
pub(crate) struct Block {
    //     id: NameId,
    pub(crate) ops: Vec<Op>,
}

#[derive(Debug, New)]
pub(crate) struct BinOp {
    pub(crate) kind: BinOpKind,
}

#[derive(Debug)]
pub(crate) enum BinOpKind {
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
    Pow,
    Sub,
    Xor,
    Mut,
}

#[derive(Debug, New)]
pub(crate) struct UnOp {
    pub(crate) kind: UnOpKind,
}

#[derive(Debug)]
pub(crate) enum UnOpKind {
    Neg,
    Not,
}

#[derive(Debug)]
pub(crate) enum ConstKind {
    Bool(bool),
    Char(char),
    Bf16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    Fun(Path),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Time(Duration),
    Unit,
}
