#![allow(clippy::useless_format)]
use crate::compiler::hir;
use crate::compiler::hir::{Name, Path};
use crate::compiler::info;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::info::paths::PathId;

use std::fmt::{self, Display, Formatter};

#[derive(From, Copy, Clone)]
pub(crate) struct State<'i> {
    info: &'i Info,
}

pub(crate) fn pretty<'i, 'j, Node>(node: &'i Node, info: &'j Info) -> Pretty<'i, Node, State<'j>> {
    node.to_pretty(State::from(info))
}

impl<'i> Display for Pretty<'i, mlir::MLIR, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(mlir, ctx) = self;
        write!(
            f,
            "module @toplevel {{{}{s0}}}",
            mlir.items
                .iter()
                .filter_map(|x| mlir.defs.get(x))
                .map_pretty(|i, f| write!(f, "{}{}", ctx.indent(), i.pretty(ctx)), ""),
            s0 = ctx,
        )?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, mlir::Item, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        match &item.kind {
            mlir::ItemKind::Fun(item)   => write!(f, "{}", item.pretty(ctx)),
            mlir::ItemKind::Enum(item)  => write!(f, "{}", item.pretty(ctx)),
            mlir::ItemKind::Task(item)  => write!(f, "{}", item.pretty(ctx)),
            mlir::ItemKind::State(item) => write!(f, "{}", item.pretty(ctx)),
        }
    }
}

impl<'i> Display for Pretty<'i, mlir::Fun, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "func @{id}({params}) -> {ty} {body}",
            id = item.name.pretty(ctx),
            params = item.params.iter().map_pretty(
                |x, f| write!(f, "{}: {}", x.pretty(ctx), x.tv.pretty(ctx)),
                ", "
            ),
            ty = item.tv.pretty(ctx),
            body = item.body.pretty(&ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, Name, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(name, ctx) = self;
        write!(f, "{}", ctx.state.info.names.resolve(name.id))
    }
}

impl<'i> Display for Pretty<'i, mlir::State, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "state {name}: {ty} = {init};",
            name = item.name.pretty(ctx),
            ty = item.tv.pretty(ctx),
            init = item.init.pretty(ctx)
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Alias, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "type {id} = {ty}",
            id = item.name.pretty(ctx),
            ty = item.tv.pretty(ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, TypeId, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(tv, ctx) = self;
        write!(f, "{}", ctx.state.info.types.resolve(**tv).pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, mlir::Task, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "task {name}({params}) ({iports}) -> ({oports}) {{{items}{s0}{s0}}}",
            name = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            iports = item.iports.pretty(ctx),
            oports = item.oports.pretty(ctx),
            items = item.items.iter().map_pretty(
                |i, f| write!(f, "{s0}{s0}{}", i.pretty(ctx.indent()), s0 = ctx.indent()),
                ""
            ),
            s0 = ctx,
        )
    }
}

impl<'i> Display for Pretty<'i, Path, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, ctx) = self;
        let buf = ctx.state.info.paths.resolve(path.id);
        write!(f, "{}", path.id.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, PathId, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(mut path, ctx) = self;
        let path = ctx.state.info.paths.resolve(*path);
        if let Some(id) = path.pred {
            write!(f, "{}_", id.pretty(ctx))?;
        }
        write!(f, "{}", path.name.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, mlir::Enum, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "enum {id} of {variants}",
            id = item.name.pretty(ctx),
            variants = item
                .variants
                .iter()
                .map_pretty(|v, f| write!(f, "{}| {}", ctx.indent(), v.pretty(ctx)), "")
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Variant, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(variant, ctx) = self;
        write!(
            f,
            "{}({})",
            variant.name.pretty(ctx),
            variant.tv.pretty(ctx)
        )
    }
}

impl<'i> Display for Pretty<'i, mlir::Var, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(var, ctx) = self;
        write!(f, "%{}", var.name.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, mlir::Op, State<'_>> {
    #[allow(clippy::many_single_char_names)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, ctx) = self;
        if let Some(var) = &op.var {
            write!(
                f,
                "{var} = {kind} {ty}",
                var = var.pretty(ctx),
                kind = op.kind.pretty(ctx),
                ty = var.tv.pretty(ctx)
            )
        } else {
            write!(f, "{kind}", kind = op.kind.pretty(ctx),)
        }
    }
}

#[rustfmt::skip]
impl<'i> Display for Pretty<'i, mlir::OpKind, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, ctx) = self;
        use mlir::{BinOpKind::*, ConstKind::*, OpKind};
        match kind {
            mlir::OpKind::Const(c) => match c {
                mlir::ConstKind::Bool(true)  => write!(f, r#"constant 1 :"#),
                mlir::ConstKind::Bool(false) => write!(f, r#"constant 0 :"#),
                mlir::ConstKind::F32(l)      => write!(f, r#"constant {}"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::F64(l)      => write!(f, r#"constant {}"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::I8(v)       => write!(f, r#"constant {} :"#, v),
                mlir::ConstKind::I16(v)      => write!(f, r#"constant {} :"#, v),
                mlir::ConstKind::I32(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I64(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U8(v)       => write!(f, r#"constant {} :"#, v),
                mlir::ConstKind::U16(v)      => write!(f, r#"constant {} :"#, v),
                mlir::ConstKind::U32(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U64(v)      => write!(f, r#"arc.constant {} :"#, v),
                mlir::ConstKind::Fun(x)      => write!(f, r#"constant {} :"#, x.pretty(ctx)),
                mlir::ConstKind::Char(_)     => todo!(),
                mlir::ConstKind::Time(_)     => todo!(),
                mlir::ConstKind::Unit        => write!(f, r#"arc.constant unit :"#),
            },
            mlir::OpKind::BinOp(tv, l, op, r) => {
                let ty = ctx.state.info.types.resolve(*tv);
                let l = l.pretty(ctx);
                let r = r.pretty(ctx);
                use hir::ScalarKind::*;
                use hir::TypeKind::*;
                match (&op.kind, ty.kind) {
                    // Add
                    (Add, Scalar(I8)) => write!(f, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    (Add, Scalar(I16)) => write!(f, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    (Add, Scalar(I32)) => write!(f, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    (Add, Scalar(I64)) => write!(f, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    (Add, Scalar(F32)) => write!(f, r#"std.addf {l}, {r} :"#, l = l, r = r),
                    (Add, Scalar(F64)) => write!(f, r#"std.addf {l}, {r} :"#, l = l, r = r),
                    // Sub
                    (Sub, Scalar(I8)) => write!(f, r#"arc.subi {l},{r} :"#, l = l, r = r),
                    (Sub, Scalar(I16)) => write!(f, r#"arc.subi {l},{r} :"#, l = l, r = r),
                    (Sub, Scalar(I32)) => write!(f, r#"arc.subi {l},{r} :"#, l = l, r = r),
                    (Sub, Scalar(I64)) => write!(f, r#"arc.subi {l},{r} :"#, l = l, r = r),
                    (Sub, Scalar(F32)) => write!(f, r#"std.subf {l},{r} :"#, l = l, r = r),
                    (Sub, Scalar(F64)) => write!(f, r#"std.subf {l},{r} :"#, l = l, r = r),
                    // Mul
                    (Mul, Scalar(I8)) => write!(f, r#"arc.muli {l},{r} :"#, l = l, r = r),
                    (Mul, Scalar(I16)) => write!(f, r#"arc.muli {l},{r} :"#, l = l, r = r),
                    (Mul, Scalar(I32)) => write!(f, r#"arc.muli {l},{r} :"#, l = l, r = r),
                    (Mul, Scalar(I64)) => write!(f, r#"arc.muli {l},{r} :"#, l = l, r = r),
                    (Mul, Scalar(F32)) => write!(f, r#"std.mulf {l},{r} :"#, l = l, r = r),
                    (Mul, Scalar(F64)) => write!(f, r#"std.mulf {l},{r} :"#, l = l, r = r),
                    // Div
                    (Div, Scalar(I8)) => write!(f, r#"arc.divi {l},{r} :"#, l = l, r = r),
                    (Div, Scalar(I16)) => write!(f, r#"arc.divi {l},{r} :"#, l = l, r = r),
                    (Div, Scalar(I32)) => write!(f, r#"arc.divi {l},{r} :"#, l = l, r = r),
                    (Div, Scalar(I64)) => write!(f, r#"arc.divi {l},{r} :"#, l = l, r = r),
                    (Div, Scalar(F32)) => write!(f, r#"std.divf {l},{r} :"#, l = l, r = r),
                    (Div, Scalar(F64)) => write!(f, r#"std.divf {l},{r} :"#, l = l, r = r),
                    // Div
                    (Pow, Scalar(I8)) => write!(f, r#"arc.powi {l},{r} :"#, l = l, r = r),
                    (Pow, Scalar(I16)) => write!(f, r#"arc.powi {l},{r} :"#, l = l, r = r),
                    (Pow, Scalar(I32)) => write!(f, r#"arc.powi {l},{r} :"#, l = l, r = r),
                    (Pow, Scalar(I64)) => write!(f, r#"arc.powi {l},{r} :"#, l = l, r = r),
                    (Pow, Scalar(F32)) => write!(f, r#"std.powf {l},{r} :"#, l = l, r = r),
                    (Pow, Scalar(F64)) => write!(f, r#"std.powf {l},{r} :"#, l = l, r = r),
                    // Lt
                    (Lt, Scalar(I8)) => write!(f, r#"arc.cmpi "lt", {l}, {r} :"#, l = l, r = r),
                    (Lt, Scalar(I16)) => write!(f, r#"arc.cmpi "lt", {l}, {r} :"#, l = l, r = r),
                    (Lt, Scalar(I32)) => write!(f, r#"arc.cmpi "lt", {l}, {r} :"#, l = l, r = r),
                    (Lt, Scalar(I64)) => write!(f, r#"arc.cmpi "lt", {l}, {r} :"#, l = l, r = r),
                    (Lt, Scalar(F32)) => write!(f, r#"std.cmpf "olt", {l}, {r} :"#, l = l, r = r),
                    (Lt, Scalar(F64)) => write!(f, r#"std.cmpf "olt", {l}, {r} :"#, l = l, r = r),
                    // Leq
                    (Leq, Scalar(I8)) => write!(f, r#"arc.cmpi "le", {l}, {r} :"#, l = l, r = r),
                    (Leq, Scalar(I16)) => write!(f, r#"arc.cmpi "le", {l}, {r} :"#, l = l, r = r),
                    (Leq, Scalar(I32)) => write!(f, r#"arc.cmpi "le", {l}, {r} :"#, l = l, r = r),
                    (Leq, Scalar(I64)) => write!(f, r#"arc.cmpi "le", {l}, {r} :"#, l = l, r = r),
                    (Leq, Scalar(F32)) => write!(f, r#"std.cmpf "ole", {l}, {r} :"#, l = l, r = r),
                    (Leq, Scalar(F64)) => write!(f, r#"std.cmpf "ole", {l}, {r} :"#, l = l, r = r),
                    // Gt
                    (Gt, Scalar(I8)) => write!(f, r#"arc.cmpi "gt", {l}, {r} :"#, l = l, r = r),
                    (Gt, Scalar(I16)) => write!(f, r#"arc.cmpi "gt", {l}, {r} :"#, l = l, r = r),
                    (Gt, Scalar(I32)) => write!(f, r#"arc.cmpi "gt", {l}, {r} :"#, l = l, r = r),
                    (Gt, Scalar(I64)) => write!(f, r#"arc.cmpi "gt", {l}, {r} :"#, l = l, r = r),
                    (Gt, Scalar(F32)) => write!(f, r#"std.cmpf "ogt", {l}, {r} :"#, l = l, r = r),
                    (Gt, Scalar(F64)) => write!(f, r#"std.cmpf "ogt", {l}, {r} :"#, l = l, r = r),
                    // Geq
                    (Geq, Scalar(I8)) => write!(f, r#"arc.cmpi "ge", {l}, {r} :"#, l = l, r = r),
                    (Geq, Scalar(I16)) => write!(f, r#"arc.cmpi "ge", {l}, {r} :"#, l = l, r = r),
                    (Geq, Scalar(I32)) => write!(f, r#"arc.cmpi "ge", {l}, {r} :"#, l = l, r = r),
                    (Geq, Scalar(I64)) => write!(f, r#"arc.cmpi "ge", {l}, {r} :"#, l = l, r = r),
                    (Geq, Scalar(F32)) => write!(f, r#"std.cmpf "oge", {l}, {r} :"#, l = l, r = r),
                    (Geq, Scalar(F64)) => write!(f, r#"std.cmpf "oge", {l}, {r} :"#, l = l, r = r),
                    // Equ
                    (Equ, Scalar(I8)) => write!(f, r#"arc.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    (Equ, Scalar(I16)) => write!(f, r#"arc.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    (Equ, Scalar(I32)) => write!(f, r#"arc.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    (Equ, Scalar(I64)) => write!(f, r#"arc.cmpi "eq", {l}, {r} :"#, l = l, r = r),
                    (Equ, Scalar(F32)) => write!(f, r#"std.cmpf "oeq", {l}, {r} :"#, l = l, r = r),
                    (Equ, Scalar(F64)) => write!(f, r#"std.cmpf "oeq", {l}, {r} :"#, l = l, r = r),
                    // Neq
                    (Neq, Scalar(I8)) => write!(f, r#"arc.cmpi "ne", {l}, {r} :"#, l = l, r = r),
                    (Neq, Scalar(I16)) => write!(f, r#"arc.cmpi "ne", {l}, {r} :"#, l = l, r = r),
                    (Neq, Scalar(I32)) => write!(f, r#"arc.cmpi "ne", {l}, {r} :"#, l = l, r = r),
                    (Neq, Scalar(I64)) => write!(f, r#"arc.cmpi "ne", {l}, {r} :"#, l = l, r = r),
                    (Neq, Scalar(F32)) => write!(f, r#"std.cmpf "one", {l}, {r} :"#, l = l, r = r),
                    (Neq, Scalar(F64)) => write!(f, r#"std.cmpf "one", {l}, {r} :"#, l = l, r = r),
                    // And
                    (And, _) => write!(f, r#"arc.and {l}, {r} :"#, l = l, r = r),
                    // Or
                    (Or, _) => write!(f, r#"arc.or  {l}, {r} :"#, l = l, r = r),
                    _ => unreachable!(),
                }
            }
            mlir::OpKind::Array(_) => todo!(),
            mlir::OpKind::Struct(xfs) => write!(
                f,
                r#""arc.make_struct"({xfs} : {ts})" :"#,
                xfs = xfs.values().all_pretty(", ", ctx),
                ts = xfs
                    .values()
                    .map_pretty(|v, f| write!(f, "{}", v.tv.pretty(ctx)), ", "),
            ),
            mlir::OpKind::Enwrap(x0, x1) => write!(
                f,
                r#""arc.enwrap"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(ctx),
                x0.pretty(ctx),
                x1.tv.pretty(ctx),
            ),
            mlir::OpKind::Unwrap(x0, x1) => write!(
                f,
                r#""arc.unwrap"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(ctx),
                x0.pretty(ctx),
                x1.tv.pretty(ctx),
            ),
            mlir::OpKind::Is(x0, x1) => write!(
                f,
                r#""arc.is"{} {{ variant = {} }} : {} ->"#,
                x1.pretty(ctx),
                x0.pretty(ctx),
                x1.tv.pretty(ctx),
            ),
            mlir::OpKind::Tuple(xs) => write!(
                f,
                r#""arc.make_tuple"({xs}) : ({ts}) ->"#,
                xs = xs.all_pretty(", ", ctx),
                ts = xs.map_pretty(|x, f| write!(f, "{}", x.tv.pretty(ctx)), ", ")
            ),
            mlir::OpKind::UnOp(_, _) => todo!(),
            mlir::OpKind::If(x, r0, r1) => write!(
                f,
                r#""arc.if"({x}) ({r0},{r1}) : i1 ->"#,
                x = x.pretty(ctx),
                r0 = r0.pretty(ctx),
                r1 = r1.pretty(ctx),
            ),
            mlir::OpKind::Emit(_) => todo!(),
            mlir::OpKind::Loop(_) => todo!(),
            mlir::OpKind::Call(x, xs) => write!(
                f,
                r#"call {callee}({args}) : ({tys}) ->"#,
                callee = x.pretty(ctx),
                args = xs.iter().all_pretty(", ", ctx),
                tys = xs
                    .iter()
                    .map_pretty(|x, f| write!(f, "{}", x.tv.pretty(ctx)), ", ")
            ),
            mlir::OpKind::CallIndirect(x, xs) => write!(
                f,
                r#"call_indirect {callee}({args}) : ({tys}) ->"#,
                callee = x.pretty(ctx),
                args = xs.iter().all_pretty(", ", ctx),
                tys = xs
                    .iter()
                    .map_pretty(|x, f| write!(f, "{}", x.tv.pretty(ctx)), ", "),
            ),
            mlir::OpKind::Return(x) => write!(
                f,
                r#"return {x} : {t}{s0}"#,
                x = x.pretty(ctx),
                t = x.tv.pretty(ctx),
                s0 = ctx
            ),
            mlir::OpKind::Res(x) => write!(
                f,
                r#""arc.block.result"({x}) : {t} -> (){s0}"#,
                x = x.pretty(ctx),
                t = x.tv.pretty(ctx),
                s0 = ctx
            ),
            mlir::OpKind::Access(x, i) => write!(
                f,
                r#""arc.struct_access"({x}) {{ field = "{i}" }} : {t}"#,
                x = x.pretty(ctx),
                i = i.pretty(ctx),
                t = x.tv.pretty(ctx)
            ),
            mlir::OpKind::Project(x, i) => write!(
                f,
                r#""arc.index_tuple"({x}) {{ index = {i} }} : {t} ->"#,
                x = x.pretty(ctx),
                t = x.tv.pretty(ctx),
                i = i,
            ),
            mlir::OpKind::Break => todo!(),
            mlir::OpKind::Log(_) => todo!(),
            mlir::OpKind::Edge((x0, p0), (x1, p1)) => {
                write!(
                    f,
                    r#""arc.edge({x0}, {x1})" {{ source_port = {p0}, target_port = {p1}}}"#,
                    x0 = x0.pretty(ctx),
                    x1 = x1.pretty(ctx),
                    p0 = p0,
                    p1 = p1
                )
            }
            mlir::OpKind::Node(x, xs) => write!(
                f,
                r#""call {task}({args})""#,
                task = x.pretty(ctx),
                args = xs.all_pretty(", ", ctx)
            ),
        }
    }
}

impl<'i> Display for Pretty<'i, mlir::Region, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(r, ctx) = self;
        write!(f, "{{{}}}", r.blocks.iter().all_pretty("", ctx.indent()))
    }
}

impl<'i> Display for Pretty<'i, mlir::Block, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(
            f,
            "{}",
            b.ops
                .iter()
                .map_pretty(|op, f| write!(f, "{}{}", ctx.indent(), op.pretty(ctx)), "")
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Type, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, ctx) = self;
        use hir::{TypeKind::*, ScalarKind::*};
        match &ty.kind {
            Scalar(kind) => match kind {
                I8   => write!(f, "i8"),
                I16  => write!(f, "i16"),
                I32  => write!(f, "i32"),
                I64  => write!(f, "i64"),
                U8   => write!(f, "u8"),
                U16  => write!(f, "u16"),
                U32  => write!(f, "u32"),
                U64  => write!(f, "u64"),
                F32  => write!(f, "f32"),
                F64  => write!(f, "f64"),
                Bool => write!(f, "i1"),
                Null => todo!(),
                Str  => todo!(),
                Unit => write!(f, "i0"),
                Char => todo!(),
                Bot  => unreachable!(),
            }
            Struct(fs)     => {
                    write!(f, "!arc.struct<{fs}>",
                        fs = fs.map_pretty(|(x, e), f| write!(f, "{} : {}", x.pretty(ctx), e.pretty(ctx)), ", "))
            }
            Nominal(x)     => write!(f, "{}", x.pretty(ctx)),
            Array(ty, sh)  => todo!(),
            Stream(ty)     => todo!(),
            Map(ty0, ty1)  => todo!(),
            Set(ty)        => todo!(),
            Vector(ty)     => todo!(),
            Tuple(tys)     => write!(f, "tuple<{tys}>", tys = tys.all_pretty(", ", ctx)),
            Optional(ty)   => todo!(),
            Fun(tys, ty)   => write!(f, "({tys}) -> {ty}", tys = tys.all_pretty(", ", ctx), ty = ty.pretty(ctx)),
            Task(ty0, ty1) => todo!(),
            Unknown        => unreachable!(),
            Err            => unreachable!(),
        }
    }
}
