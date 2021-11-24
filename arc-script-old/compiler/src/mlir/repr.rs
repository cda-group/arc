use crate::ast;
use crate::hir;
pub(crate) use crate::hir::Name;
pub(crate) use crate::hir::Path;
pub(crate) use crate::hir::PathId;
pub(crate) use crate::hir::Type;
pub(crate) use crate::hir::TypeKind;
pub(crate) use crate::hir::ScalarKind;

use crate::info::files::Loc;

use arc_script_compiler_shared::New;
use arc_script_compiler_shared::OrdMap;
use arc_script_compiler_shared::VecMap;

use time::Duration;

/// MLIR is a Multi-Level Intermediate Representation which is the final
/// representation within the Arc-Script compiler. In contrast to the HIR,
/// MLIR is in SSA form.
#[derive(Debug, New, Default)]
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
    ExternFun(ExternFun),
    ExternType(ExternType),
}

#[derive(New, Debug)]
pub(crate) struct Fun {
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) body: Block,
    pub(crate) t: Type,
}

#[derive(New, Debug)]
pub(crate) struct ExternFun {
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) rt: Type,
}

#[derive(New, Debug)]
pub(crate) struct ExternType {
    pub(crate) path: Path,
    pub(crate) params: Vec<Param>,
    pub(crate) items: Vec<PathId>,
}

#[derive(New, Debug)]
pub(crate) struct Task {
    /// Path of the task
    pub(crate) path: Path,
    /// Parameters for instantiating the task
    pub(crate) params: Vec<Param>,
    /// Input stream typess
    pub(crate) istream_ts: Vec<Type>,
    /// Output stream types
    pub(crate) ostream_ts: Vec<Type>,
    /// Type of the task (struct of parameters)
    pub(crate) this_t: Type,
    /// Task I/O
    pub(crate) ievent: Param,
    pub(crate) oevent_t: Type,
    /// Event handler.
    pub(crate) on_event: Block,
    /// Statements run at startup.
    pub(crate) on_start: Block,
}

#[derive(New, Debug, Copy, Clone)]
pub(crate) struct Var {
    pub(crate) kind: VarKind,
    pub(crate) scope: ScopeKind,
    pub(crate) t: Type,
}

#[derive(Debug, Clone, Copy, New)]
pub(crate) struct Param {
    pub(crate) kind: VarKind,
    pub(crate) t: Type,
}

pub(crate) use hir::ScopeKind;

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
    pub(crate) param: Param,
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
    Panic,
    Noop,
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
