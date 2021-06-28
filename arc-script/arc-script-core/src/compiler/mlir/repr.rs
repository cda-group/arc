use crate::compiler::ast;
pub(crate) use crate::compiler::hir::Name;
pub(crate) use crate::compiler::hir::Path;
pub(crate) use crate::compiler::hir::PathId;
pub(crate) use crate::compiler::hir::Type;
pub(crate) use crate::compiler::hir::TypeKind;
pub(crate) use crate::compiler::hir::ScalarKind;

use crate::compiler::info::files::Loc;

use arc_script_core_shared::New;
use arc_script_core_shared::OrdMap;
use arc_script_core_shared::VecMap;

use half::bf16;
use half::f16;
use time::Duration;

/// MLIR is a Multi-Level Intermediate Representation which is the final
/// representation within the Arc-Script compiler. In contrast to the HIR,
/// MLIR is in SSA form.
#[derive(Debug, New)]
pub(crate) struct MLIR {
    /// Top-level items
    pub(crate) items: Vec<PathId>,
    /// Definitions of items.
    pub(crate) defs: OrdMap<PathId, Item>,
}

#[derive(New, Debug)]
pub(crate) struct Item {
    pub(crate) kind: ItemKind,
    pub(crate) loc: Loc,
}

#[derive(Debug)]
pub(crate) enum ItemKind {
    Enum(Enum),
    Fun(Fun),
    Task(Task),
}

#[derive(New, Debug)]
pub(crate) struct Fun {
    pub(crate) path: Path,
    pub(crate) params: Vec<Var>,
    pub(crate) body: Block,
    pub(crate) t: Type,
}

#[derive(New, Debug, Copy, Clone)]
pub(crate) struct Var {
    pub(crate) kind: VarKind,
    pub(crate) t: Type,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum VarKind {
    Ok(Name),
    Elided,
}

#[derive(New, Debug)]
pub(crate) struct Enum {
    pub(crate) path: Path,
    pub(crate) variants: Vec<Variant>,
}

#[derive(New, Debug)]
pub(crate) struct Variant {
    pub(crate) path: Path,
    pub(crate) t: Type,
    pub(crate) loc: Loc,
}

/// A task is a generic low-level primitive which resembles a node in the dataflow graph.
#[derive(New, Debug)]
pub(crate) struct Task {
    pub(crate) path: Path,
    /// Type of the task.
    pub(crate) t: Type,
    /// Side-input parameters to the task.
    pub(crate) params: Vec<Var>,
    /// Input ports to the task.
    pub(crate) iports: Type,
    /// Output ports of the task.
    pub(crate) oports: Type,
    /// Event handler.
    pub(crate) handler: Op,
    /// Items of the task.
    pub(crate) items: Vec<Path>,
}

#[derive(New, Debug)]
pub(crate) struct Setting {
    pub(crate) kind: SettingKind,
    pub(crate) loc: Loc,
}

#[derive(Debug)]
pub(crate) enum SettingKind {
    Activate(ast::Name),
    Calibrate(ast::Name, ConstKind),
}

#[derive(Debug, New)]
pub(crate) struct Op {
    pub(crate) var: Var,
    pub(crate) kind: OpKind,
    pub(crate) loc: Loc,
}

#[derive(Debug)]
pub(crate) enum OpKind {
    Access(Var, Name),
    Array(Vec<Var>),
    BinOp(Var, BinOp, Var),
    Break(Var),
    Continue,
    Call(Path, Vec<Var>),
    CallIndirect(Var, Vec<Var>),
    CallMethod(Var, Name, Vec<Var>),
    Const(ConstKind),
    Emit(Var),
    If(Var, Block, Block),
    Log(Var),
    Loop(Var),
    Project(Var, usize),
    Result(Var),
    Return(Var),
    Struct(VecMap<Name, Var>),
    Tuple(Vec<Var>),
    UnOp(UnOp, Var),
    Enwrap(Path, Var),
    Unwrap(Path, Var),
    Is(Path, Var),
}

#[derive(Debug, New)]
pub(crate) struct Block {
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
    Mut,
    Neq,
    Or,
    Pow,
    Sub,
    Xor,
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
    Noop,
}

impl OpKind {
    pub(crate) const fn get_type_specifier(&self, t: Type) -> Type {
        match self {
            OpKind::BinOp(l, op, _) => {
                use BinOpKind::*;
                match op.kind {
                    Equ | Geq | Gt | Leq | Lt | Neq => l.t,
                    _ => t,
                }
            }
            _ => t,
        }
    }
}
