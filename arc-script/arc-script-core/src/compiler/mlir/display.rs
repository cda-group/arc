#![allow(clippy::useless_format)]
use crate::compiler::hir;
use crate::compiler::hir::{Name, Path};

use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::pretty::*;
use arc_script_core_shared::From;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

#[derive(From, Copy, Clone)]
pub(crate) struct Context<'i> {
    info: &'i Info,
}

pub(crate) fn pretty<'i, 'j, Node>(
    node: &'i Node,
    info: &'j Info,
) -> Pretty<'i, Node, Context<'j>> {
    node.to_pretty(Context::from(info))
}

impl<'i> Display for Pretty<'i, mlir::MLIR, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(mlir, fmt) = self;
        write!(
            f,
            "module @toplevel {{{}{s0}}}",
            mlir.items
                .iter()
                .filter_map(|x| mlir.defs.get(x))
                .map_pretty(|i, f| write!(f, "{}{}", fmt.indent(), i.pretty(fmt)), ""),
            s0 = fmt,
        )?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, mlir::Item, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match &item.kind {
            mlir::ItemKind::Fun(item)   => write!(f, "{}", item.pretty(fmt)),
            mlir::ItemKind::Enum(item)  => write!(f, "{}", item.pretty(fmt)),
            mlir::ItemKind::Task(item)  => write!(f, "{}", item.pretty(fmt)),
            mlir::ItemKind::State(item) => write!(f, "{}", item.pretty(fmt)),
        }
    }
}

impl<'i> Display for Pretty<'i, mlir::Fun, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "func @{id}({params}) -> {ty} {body}",
            id = item.path.pretty(fmt),
            params = item.params.iter().map_pretty(
                |x, f| write!(f, "{}: {}", x.pretty(fmt), x.tv.pretty(fmt)),
                ", "
            ),
            ty = item.tv.pretty(fmt),
            body = item.body.pretty(&fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, Name, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(name, fmt) = self;
        write!(f, "{}", fmt.ctx.info.names.resolve(name.id))
    }
}

impl<'i> Display for Pretty<'i, mlir::State, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "state {name}: {ty} = {init};",
            name = item.path.pretty(fmt),
            ty = item.tv.pretty(fmt),
            init = item.init.pretty(fmt)
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Alias, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "type {id} = {ty}",
            id = item.path.pretty(fmt),
            ty = item.tv.pretty(fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, TypeId, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(tv, fmt) = self;
        write!(f, "{}", fmt.ctx.info.types.resolve(**tv).pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, mlir::Task, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "task {name}({params}) ({iports}) -> ({oports}) {{{items}{s0}{s0}}}",
            name = item.path.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            iports = item.iports.pretty(fmt),
            oports = item.oports.pretty(fmt),
            items = item.items.iter().map_pretty(
                |i, f| write!(f, "{s0}{s0}{}", i.pretty(fmt.indent()), s0 = fmt.indent()),
                ""
            ),
            s0 = fmt,
        )
    }
}

impl<'i> Display for Pretty<'i, Path, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, fmt) = self;
        let _buf = fmt.ctx.info.paths.resolve(path.id);
        write!(f, "{}", path.id.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, PathId, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, fmt) = self;
        let path = fmt.ctx.info.paths.resolve(*path);
        if let Some(id) = path.pred {
            write!(f, "{}_", id.pretty(fmt))?;
        }
        write!(f, "{}", path.name.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, mlir::Enum, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        // TODO: Find way to represent enums in MLIR
        write!(
            f,
            "arc.enum<{id}> of {variants}",
            id = item.path.pretty(fmt),
            variants = item
                .variants
                .iter()
                .map_pretty(|v, f| write!(f, "{}| {}", fmt.indent(), v.pretty(fmt)), "")
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Variant, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(variant, fmt) = self;
        write!(
            f,
            "{}({})",
            variant.path.pretty(fmt),
            variant.tv.pretty(fmt)
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Var, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(var, fmt) = self;
        write!(f, "%{}", var.name.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, mlir::Op, Context<'_>> {
    #[allow(clippy::many_single_char_names)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, fmt) = self;
        use mlir::ConstKind::*;
        use mlir::OpKind::*;
        match op.var {
            Some(var) if matches!(op.kind, Const(Bool(_))) => {
                write!(
                    f,
                    "{var} = {kind}",
                    var = var.pretty(fmt),
                    kind = op.kind.pretty(fmt),
                )
            }
            Some(var) => {
                write!(
                    f,
                    "{var} = {kind} {ty}",
                    var = var.pretty(fmt),
                    kind = op.kind.pretty(fmt),
                    ty = var.tv.pretty(fmt)
                )
            }
            None => write!(f, "{kind}", kind = op.kind.pretty(fmt)),
        }
    }
}

#[rustfmt::skip]
impl<'i> Display for Pretty<'i, mlir::OpKind, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, fmt) = self;
        use mlir::BinOpKind::*;
        match kind {
            mlir::OpKind::Const(c) => match c {
                mlir::ConstKind::Bool(true)  => write!(f, r#"constant true"#),
                mlir::ConstKind::Bool(false) => write!(f, r#"constant false"#),
                mlir::ConstKind::Bf16(l)     => write!(f, r#"constant {} :"#, ryu::Buffer::new().format(l.to_f32())),
                mlir::ConstKind::F16(l)      => write!(f, r#"constant {} :"#, ryu::Buffer::new().format(l.to_f32())),
                mlir::ConstKind::F32(l)      => write!(f, r#"constant {} :"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::F64(l)      => write!(f, r#"constant {} :"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::I8(v)       => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I16(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I32(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I64(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U8(v)       => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U16(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U32(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U64(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::Fun(x)      => write!(f, r#"constant {} :"#, x.pretty(fmt)),
                mlir::ConstKind::Char(_)     => todo!(),
                mlir::ConstKind::Time(_)     => todo!(),
                mlir::ConstKind::Unit        => todo!(),
            },
            mlir::OpKind::BinOp(tv, l, op, r) => {
                let l = l.pretty(fmt);
                let r = r.pretty(fmt);
                let info = fmt.ctx.info;
                use hir::ScalarKind::*;
                use hir::TypeKind::*;
                match &op.kind {
                    Add  if tv.is_int(info)   => write!(f, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    Add  if tv.is_float(info) => write!(f, r#"std.addf {l}, {r} :"#, l = l, r = r),
                    Sub  if tv.is_int(info)   => write!(f, r#"arc.subi {l}, {r} :"#, l = l, r = r),
                    Sub  if tv.is_float(info) => write!(f, r#"std.subf {l}, {r} :"#, l = l, r = r),
                    Mul  if tv.is_int(info)   => write!(f, r#"arc.muli {l}, {r} :"#, l = l, r = r),
                    Mul  if tv.is_float(info) => write!(f, r#"std.mulf {l}, {r} :"#, l = l, r = r),
                    Div  if tv.is_int(info)   => write!(f, r#"arc.divi {l}, {r} :"#, l = l, r = r),
                    Div  if tv.is_float(info) => write!(f, r#"std.divf {l}, {r} :"#, l = l, r = r),
                    Mod  if tv.is_int(info)   => write!(f, r#"arc.modf {l}, {r} :"#, l = l, r = r),
                    Mod  if tv.is_float(info) => write!(f, r#"std.modf {l}, {r} :"#, l = l, r = r),
                    Pow  if tv.is_int(info)   => write!(f, r#"arc.powi {l}, {r} :"#, l = l, r = r),
                    Pow  if tv.is_float(info) => write!(f, r#"std.powf {l}, {r} :"#, l = l, r = r),
                    Lt   if tv.is_int(info)   => write!(f, r#"arc.cmpi "lt", {l}, {r} :"#, l = l, r = r),
                    Lt   if tv.is_float(info) => write!(f, r#"std.cmpf "olt", {l}, {r} :"#, l = l, r = r),
                    Leq  if tv.is_int(info)   => write!(f, r#"arc.cmpi "le", {l}, {r} :"#, l = l, r = r),
                    Leq  if tv.is_float(info) => write!(f, r#"std.cmpf "ole", {l}, {r} :"#, l = l, r = r),
                    Gt   if tv.is_int(info)   => write!(f, r#"arc.cmpi "gt", {l}, {r} :"#, l = l, r = r),
                    Gt   if tv.is_float(info) => write!(f, r#"std.cmpf "ogt", {l}, {r} :"#, l = l, r = r),
                    Geq  if tv.is_int(info)   => write!(f, r#"arc.cmpi "ge", {l}, {r} :"#, l = l, r = r),
                    Geq  if tv.is_float(info) => write!(f, r#"std.cmpf "oge", {l}, {r} :"#, l = l, r = r),
                    Equ  if tv.is_int(info)   => write!(f, r#"arc.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    Equ  if tv.is_float(info) => write!(f, r#"std.cmpf "oeq", {l}, {r} :"#, l = l, r = r),
                    Equ  if tv.is_bool(info)  => write!(f, r#"std.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    Neq  if tv.is_int(info)   => write!(f, r#"arc.cmpi "ne", {l}, {r} :"#, l = l, r = r),
                    Neq  if tv.is_float(info) => write!(f, r#"std.cmpf "one", {l}, {r} :"#, l = l, r = r),
                    Neq  if tv.is_bool(info)  => write!(f, r#"std.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    And  if tv.is_bool(info)  => write!(f, r#"arc.and {l}, {r} :"#, l = l, r = r),
                    Or   if tv.is_bool(info)  => write!(f, r#"arc.or {l}, {r} :"#, l = l, r = r),
                    Xor  if tv.is_bool(info)  => write!(f, r#"std.xor {l}, {r} :"#, l = l, r = r),
                    Band if tv.is_int(info)   => write!(f, r#"arc.and {l}, {r} :"#, l = l, r = r),
                    Bor  if tv.is_int(info)   => write!(f, r#"arc.or {l}, {r} :"#, l = l, r = r),
                    Bxor if tv.is_int(info)   => write!(f, r#"arc.xor {l}, {r} :"#, l = l, r = r),
                    op => {
                        let ty = info.types.resolve(*tv);
                        unreachable!("Undefined op: {:?} for {:?}", op, ty)
                    },
                }
            }
            mlir::OpKind::Array(_) => todo!(),
            mlir::OpKind::Struct(xfs) => write!(
                f,
                r#""arc.make_struct"({xfs} : {ts})" :"#,
                xfs = xfs.values().all_pretty(", ", fmt),
                ts = xfs
                    .values()
                    .map_pretty(|v, f| write!(f, "{}", v.tv.pretty(fmt)), ", "),
            ),
            mlir::OpKind::Enwrap(x0, x1) => write!(
                f,
                r#""arc.enwrap"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(fmt),
                x0.pretty(fmt),
                x1.tv.pretty(fmt),
            ),
            mlir::OpKind::Unwrap(x0, x1) => write!(
                f,
                r#""arc.unwrap"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(fmt),
                x0.pretty(fmt),
                x1.tv.pretty(fmt),
            ),
            mlir::OpKind::Is(x0, x1) => write!(
                f,
                r#""arc.is"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(fmt),
                x0.pretty(fmt),
                x1.tv.pretty(fmt),
            ),
            mlir::OpKind::Tuple(xs) => write!(
                f,
                r#""arc.make_tuple"({xs}) : ({ts}) ->"#,
                xs = xs.all_pretty(", ", fmt),
                ts = xs.map_pretty(|x, f| write!(f, "{}", x.tv.pretty(fmt)), ", ")
            ),
            mlir::OpKind::UnOp(_, _) => todo!(),
            mlir::OpKind::If(x, r0, r1) => write!(
                f,
                r#""arc.if"({x}) ({r0},{r1}) : (i1) ->"#,
                x = x.pretty(fmt),
                r0 = r0.pretty(fmt),
                r1 = r1.pretty(fmt),
            ),
            mlir::OpKind::Emit(_) => todo!(),
            mlir::OpKind::Loop(_) => todo!(),
            mlir::OpKind::Call(x, xs) => write!(
                f,
                r#"call {callee}({args}) : ({tys}) ->"#,
                callee = x.pretty(fmt),
                args = xs.iter().all_pretty(", ", fmt),
                tys = xs
                    .iter()
                    .map_pretty(|x, f| write!(f, "{}", x.tv.pretty(fmt)), ", ")
            ),
            mlir::OpKind::CallIndirect(x, xs) => write!(
                f,
                r#"call_indirect {callee}({args}) : ({tys}) ->"#,
                callee = x.pretty(fmt),
                args = xs.iter().all_pretty(", ", fmt),
                tys = xs
                    .iter()
                    .map_pretty(|x, f| write!(f, "{}", x.tv.pretty(fmt)), ", "),
            ),
            mlir::OpKind::Return(x)
		if matches!(fmt.ctx.info.types.resolve(x.tv).kind,
			    hir::TypeKind::Scalar(hir::ScalarKind::Unit)) =>
		write!(
                    f,
                    r#"return {s0}"#,
                    s0 = fmt
		),
            mlir::OpKind::Return(x) => write!(
                f,
                r#"return {x} : {t}{s0}"#,
                x = x.pretty(fmt),
                t = x.tv.pretty(fmt),
                s0 = fmt
            ),
            mlir::OpKind::Res(x) => write!(
                f,
                r#""arc.block.result"({x}) : ({t}) -> (){s0}"#,
                x = x.pretty(fmt),
                t = x.tv.pretty(fmt),
                s0 = fmt
            ),
            mlir::OpKind::Access(x, i) => write!(
                f,
                r#""arc.struct_access"({x}) {{ field = "{i}" }} : {t}"#,
                x = x.pretty(fmt),
                i = i.pretty(fmt),
                t = x.tv.pretty(fmt)
            ),
            mlir::OpKind::Project(x, i) => write!(
                f,
                r#""arc.index_tuple"({x}) {{ index = {i} }} : ({t}) ->"#,
                x = x.pretty(fmt),
                t = x.tv.pretty(fmt),
                i = i,
            ),
            mlir::OpKind::Break => todo!(),
            mlir::OpKind::Log(_) => todo!(),
            mlir::OpKind::Edge((x0, p0), (x1, p1)) => {
                write!(
                    f,
                    r#""arc.edge({x0}, {x1})" {{ source_port = {p0}, target_port = {p1}}}"#,
                    x0 = x0.pretty(fmt),
                    x1 = x1.pretty(fmt),
                    p0 = p0,
                    p1 = p1
                )
            }
            mlir::OpKind::Node(x, xs) => write!(
                f,
                r#""call {task}({args})""#,
                task = x.pretty(fmt),
                args = xs.all_pretty(", ", fmt)
            ),
        }
    }
}

impl<'i> Display for Pretty<'i, mlir::Region, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(r, fmt) = self;
        write!(f, "{{{}}}", r.blocks.iter().all_pretty("", fmt.indent()))
    }
}

impl<'i> Display for Pretty<'i, mlir::Block, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(
            f,
            "{}",
            b.ops
                .iter()
                .map_pretty(|op, f| write!(f, "{}{}", fmt.indent(), op.pretty(fmt)), "")
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Type, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, fmt) = self;
        use hir::{TypeKind::*, ScalarKind::*};
        match &ty.kind {
            Scalar(kind) => match kind {
                I8   => write!(f, "si8"),
                I16  => write!(f, "si16"),
                I32  => write!(f, "si32"),
                I64  => write!(f, "si64"),
                U8   => write!(f, "ui8"),
                U16  => write!(f, "ui16"),
                U32  => write!(f, "ui32"),
                U64  => write!(f, "ui64"),
                Bf16 => write!(f, "bf16"),
                F16  => write!(f, "f16"),
                F32  => write!(f, "f32"),
                F64  => write!(f, "f64"),
                Bool => write!(f, "i1"),
                Null => todo!(),
                Str  => todo!(),
                Unit => write!(f, "()"),
                Char => todo!(),
                Bot  => unreachable!(),
            }
            Struct(fs)     => {
                    write!(f, "!arc.struct<{fs}>",
                        fs = fs.map_pretty(|(x, e), f| write!(f, "{} : {}", x.pretty(fmt), e.pretty(fmt)), ", "))
            }
            Nominal(x)      => write!(f, "{}", x.pretty(fmt)),
            Array(_ty, _sh) => todo!(),
            Stream(_ty)     => todo!(),
            Map(_ty0, _ty1) => todo!(),
            Set(_ty)        => todo!(),
            Vector(_ty)     => todo!(),
            Tuple(tys)      => write!(f, "tuple<{tys}>", tys = tys.all_pretty(", ", fmt)),
            Optional(_ty)   => todo!(),
            Fun(tys, ty)    => write!(f, "({tys}) -> {ty}", tys = tys.all_pretty(", ", fmt), ty = ty.pretty(fmt)),
            Unknown         => unreachable!(),
            Err             => unreachable!(),
        }
    }
}
