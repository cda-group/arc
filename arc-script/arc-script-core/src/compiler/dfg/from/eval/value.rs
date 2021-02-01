use crate::compiler::dfg::from::eval::stack::Frame;
use crate::compiler::dfg::DFG;
use crate::compiler::dfg::{Node, Port};
use crate::compiler::hir::{
    BinOp, BinOpKind, BinOpKind::*, Expr, ExprKind, LitKind, Path, TypeKind, UnOp, UnOpKind,
    UnOpKind::*, HIR,
};
use crate::compiler::info::diags::Error;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::shared::{New, VecMap};

#[derive(Clone, Debug, New)]
pub(crate) struct Value {
    pub(crate) kind: ValueKind,
    pub(crate) ty: TypeId,
}

/// A struct representing the possible values an Arc-Script can evaluate into.
///
/// NB: By the time expressions are evaluated, all closures will be converted
/// into top-level items through lambda lifting.
#[derive(Clone, Debug)]
pub(crate) enum ValueKind {
    Unit,
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Char(char),
    Str(String),
    Bool(bool),
    Item(Path),
    Task(Path, Node),
    Stream(Node, Port),
    Vector(Vec<Value>),
    Tuple(Vec<Value>),
    Array(Vec<Value>),
    Variant(Path, Box<Value>),
    Struct(VecMap<NameId, Value>),
}
