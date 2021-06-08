#![allow(clippy::useless_format)]

#[path = "../pretty.rs"]
#[macro_use]
pub(crate) mod pretty;
use pretty::*;

use crate::compiler::hir;
use crate::compiler::hir::{Name, Path};

use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;
use crate::compiler::mlir;

use arc_script_core_shared::get;
use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use std::process::Command;

#[derive(New, Copy, Clone, Shrinkwrap)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i Info,
    mlir: &'i mlir::MLIR,
}

impl mlir::MLIR {
    pub(crate) fn display<'i>(&'i self, info: &'i Info) -> Pretty<'i, Self, Context<'i>> {
        self.pretty(self, info)
    }
    pub(crate) fn pretty<'i, 'j, Node>(
        &'j self,
        node: &'i Node,
        info: &'j Info,
    ) -> Pretty<'i, Node, Context<'j>> {
        node.to_pretty(Context::new(info, self))
    }
}

pub(crate) fn run_arc_mlir(infile: &std::path::Path, outfile: &std::path::Path) {
    let arc_script_bin = std::env::current_exe().unwrap();

    // We want arc-script to be able to find the arc-mlir binary
    // without it being in the path, so we look for it relative to the
    // current binary. As this won't work for the various
    // cargo/rust-based tests, we fall back to the default search path
    // when the relative lookup fails. The arc-cargo wrapper will set
    // up PATH to include the directory of the arc-mlir binary.
    let arc_mlir_bin = match arc_script_bin
        .parent()
        .unwrap()
        .join("..")
        .join("..")
        .join("bin")
        .join("arc-mlir")
        .canonicalize()
    {
        Ok(val) => val,
        Err(_) => std::path::PathBuf::from("arc-mlir"),
    };

    Command::new(arc_mlir_bin)
        .arg(infile)
        .arg("-o")
        .arg(outfile)
        .arg("-arc-to-rust")
        .arg("-inline-rust")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    Command::new("rustfmt")
        .arg(outfile)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}

pretty! {
    [node, fmt, w]

    mlir::MLIR => write!(w, "module @toplevel {{{items}{s0}}}",
        items = node.items
            .iter()
            .filter_map(|i|
            {
                 let x = node.resolve(i);
	         match x.kind {
                      // There is no need to "declare" enum
                      // types before use in MLIR, so just
                      // filter them out
                      mlir::ItemKind::Enum(_) => None,
                      _ => Some(x),
            }})
            .map_pretty(|i, w| write!(w, "{}{}", fmt.indent(), i.pretty(fmt)), ""),
        s0 = fmt,
    ),
    mlir::Item => match &node.kind {
        mlir::ItemKind::Fun(item)   => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::Enum(item)  => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::Task(item)  => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::State(item) => write!(w, "{}", item.pretty(fmt)),
    },
    mlir::Fun => write!(w, "func @{id}({params}) -> {ty} {body}",
        id = node.path.pretty(fmt),
        params = node.params.iter().map_pretty(
            |x, w| write!(w, "{}: {}", x.pretty(fmt), x.t.pretty(fmt)),
            ", "
        ),
        ty = node.t.pretty(fmt),
        body = node.body.pretty(&fmt),
    ),
    mlir::Name => write!(w, "{}", fmt.names.resolve(node)),
    mlir::State => write!(w, "state {name}: {ty} = {init};",
        name = node.path.pretty(fmt),
        ty = node.t.pretty(fmt),
        init = node.init.pretty(fmt)
    ),
    mlir::Alias => write!(w, "type {id} = {ty}",
        id = node.path.pretty(fmt),
        ty = node.t.pretty(fmt),
    ),
    mlir::Task => write!(w, "task {name}({params}) ({iports}) -> ({oports}) {{{items}{s0}{s0}}}",
        name = node.path.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        iports = node.iports.pretty(fmt),
        oports = node.oports.pretty(fmt),
        items = node.items.iter().map_pretty(
            |i, w| write!(w, "{s0}{s0}{}", i.pretty(fmt.indent()), s0 = fmt.indent()),
            ""
        ),
        s0 = fmt,
    ),
    mlir::Path => write!(w, "{}", node.id.pretty(fmt)),
    mlir::PathId => {
        let kind = fmt.paths.resolve(*node);
        if let Some(id) = kind.pred {
            write!(w, "{}_", id.pretty(fmt))?;
        }
        write!(w, "{}", kind.name.pretty(fmt))
    },
    mlir::Enum => write!(w, "!arc.enum<{}>", node.variants.iter().all_pretty(", ", fmt)),
    mlir::Variant => write!(w, "{} : {}", node.path.pretty(fmt), node.t.pretty(fmt)),
    mlir::Var => write!(w, "%{}", node.name.pretty(fmt)),
    mlir::Op => {
        use mlir::ConstKind::*;
        use mlir::OpKind::*;
        match node.var {
            Some(var) if matches!(node.kind, Const(Bool(_))) => {
                write!(
                    w,
                    "{var} = {kind}",
                    var = var.pretty(fmt),
                    kind = node.kind.pretty(fmt),
                )
            }
	    Some(var) if matches!(node.kind, Const(Unit))=> {
		write!(w, "// No value")
	    }
            Some(var) => {
		let ty = node.kind.get_type_specifier(var.t);
		match fmt.types.resolve(ty) {
		    hir::repr::TypeKind::Scalar(hir::repr::ScalarKind::Unit) =>
			write!(
			    w,
			    "{kind}",
			    kind = node.kind.pretty(fmt),
			),
		    _ =>
			write!(
			    w,
			    "{var} = {kind} {ty}",
			    var = var.pretty(fmt),
			    kind = node.kind.pretty(fmt),
			    ty = ty.pretty(fmt),
			),
		}
            }
            None => write!(w, "{kind}", kind = node.kind.pretty(fmt)),
        }
    },
    mlir::OpKind => {
        use mlir::BinOpKind::*;
        match node {
            mlir::OpKind::Const(c) => match c {
                mlir::ConstKind::Bool(true)  => write!(w, r#"constant true"#),
                mlir::ConstKind::Bool(false) => write!(w, r#"constant false"#),
                mlir::ConstKind::Bf16(l)     => write!(w, r#"constant {} :"#, ryu::Buffer::new().format(l.to_f32())),
                mlir::ConstKind::F16(l)      => write!(w, r#"constant {} :"#, ryu::Buffer::new().format(l.to_f32())),
                mlir::ConstKind::F32(l)      => write!(w, r#"constant {} :"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::F64(l)      => write!(w, r#"constant {} :"#, ryu::Buffer::new().format(*l)),
                mlir::ConstKind::I8(v)       => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I16(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I32(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::I64(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U8(v)       => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U16(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U32(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::U64(v)      => write!(w, r#"arc.constant {} :"#, v),
                mlir::ConstKind::Fun(x)      => write!(w, r#"constant @{} :"#, x.pretty(fmt)),
                mlir::ConstKind::Char(_)     => crate::todo!(),
                mlir::ConstKind::Time(_)     => crate::todo!(),
                mlir::ConstKind::Unit        => write!(w, r#"none"#),
            },
            mlir::OpKind::BinOp(t, l, op, r) => {
                let l = l.pretty(fmt);
                let r = r.pretty(fmt);
                let info = fmt.info;
                use hir::ScalarKind::*;
                use hir::TypeKind::*;
                match &op.kind {
                    Add  if t.is_int(info)   => write!(w, r#"arc.addi {l}, {r} :"#, l = l, r = r),
                    Add  if t.is_float(info) => write!(w, r#"addf {l}, {r} :"#, l = l, r = r),
                    Sub  if t.is_int(info)   => write!(w, r#"arc.subi {l}, {r} :"#, l = l, r = r),
                    Sub  if t.is_float(info) => write!(w, r#"subf {l}, {r} :"#, l = l, r = r),
                    Mul  if t.is_int(info)   => write!(w, r#"arc.muli {l}, {r} :"#, l = l, r = r),
                    Mul  if t.is_float(info) => write!(w, r#"mulf {l}, {r} :"#, l = l, r = r),
                    Div  if t.is_int(info)   => write!(w, r#"arc.divi {l}, {r} :"#, l = l, r = r),
                    Div  if t.is_float(info) => write!(w, r#"divf {l}, {r} :"#, l = l, r = r),
                    Mod  if t.is_int(info)   => write!(w, r#"arc.remi {l}, {r} :"#, l = l, r = r),
                    Mod  if t.is_float(info) => write!(w, r#"remf {l}, {r} :"#, l = l, r = r),
                    Pow  if t.is_int(info)   => write!(w, r#"arc.powi {l}, {r} :"#, l = l, r = r),
                    Pow  if t.is_float(info) => write!(w, r#"math.powf {l}, {r} :"#, l = l, r = r),
                    Lt   if t.is_int(info)   => write!(w, r#"arc.cmpi lt, {l}, {r} :"#, l = l, r = r),
                    Lt   if t.is_float(info) => write!(w, r#"cmpf olt, {l}, {r} :"#, l = l, r = r),
                    Leq  if t.is_int(info)   => write!(w, r#"arc.cmpi le, {l}, {r} :"#, l = l, r = r),
                    Leq  if t.is_float(info) => write!(w, r#"cmpf ole, {l}, {r} :"#, l = l, r = r),
                    Gt   if t.is_int(info)   => write!(w, r#"arc.cmpi gt, {l}, {r} :"#, l = l, r = r),
                    Gt   if t.is_float(info) => write!(w, r#"cmpf ogt, {l}, {r} :"#, l = l, r = r),
                    Geq  if t.is_int(info)   => write!(w, r#"arc.cmpi ge, {l}, {r} :"#, l = l, r = r),
                    Geq  if t.is_float(info) => write!(w, r#"cmpf oge, {l}, {r} :"#, l = l, r = r),
                    Equ  if t.is_int(info)   => write!(w, r#"arc.cmpi eq, {l}, {r} :"#, l = l, r = r),
                    Equ  if t.is_float(info) => write!(w, r#"cmpf oeq, {l}, {r} :"#, l = l, r = r),
                    Equ  if t.is_bool(info)  => write!(w, r#"cmpi eq, {l}, {r} :"#, l = l, r = r),
                    Neq  if t.is_int(info)   => write!(w, r#"arc.cmpi ne, {l}, {r} :"#, l = l, r = r),
                    Neq  if t.is_float(info) => write!(w, r#"cmpf one, {l}, {r} :"#, l = l, r = r),
                    Neq  if t.is_bool(info)  => write!(w, r#"cmpi ne, {l}, {r} :"#, l = l, r = r),
                    And  if t.is_bool(info)  => write!(w, r#"and {l}, {r} :"#, l = l, r = r),
                    Or   if t.is_bool(info)  => write!(w, r#"or {l}, {r} :"#, l = l, r = r),
                    Xor  if t.is_bool(info)  => write!(w, r#"xor {l}, {r} :"#, l = l, r = r),
                    Band if t.is_int(info)   => write!(w, r#"arc.and {l}, {r} :"#, l = l, r = r),
                    Bor  if t.is_int(info)   => write!(w, r#"arc.or {l}, {r} :"#, l = l, r = r),
                    Bxor if t.is_int(info)   => write!(w, r#"arc.xor {l}, {r} :"#, l = l, r = r),
                    op => {
                        let kind = info.types.resolve(*t);
                        unreachable!("Undefined op: {:?} for {:?}", op, kind)
                    },
                }
            }
            mlir::OpKind::Array(_) => todo!(),
            mlir::OpKind::Struct(xfs) => write!(
                w,
                r#"arc.make_struct({xfs} : {ts}) :"#,
                xfs = xfs.values().all_pretty(", ", fmt),
                ts = xfs
                    .values()
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
            ),
	          // %s = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32>
            mlir::OpKind::Enwrap(x0, x1) => write!(
                w,
                r#"arc.make_enum ({} : {}) as "{}" : "#,
                x1.pretty(fmt),
		            x1.t.pretty(fmt),
                x0.pretty(fmt),
		        ),
	          // %r = arc.enum_access "b" in (%e : !arc.enum<a : i32, b : f32>) : f32
            mlir::OpKind::Unwrap(x0, x1) => write!(
                w,
                r#"arc.enum_access "{}" in ({} : {}) : "#,
                x0.pretty(fmt),
                x1.pretty(fmt),
                x1.t.pretty(fmt),
            ),
		        // %r = arc.enum_check (%e : !arc.enum<a : i32, b : f32>) is "a" : i1
            mlir::OpKind::Is(x0, x1) => write!(
                w,
                r#"arc.enum_check ({} : {}) is "{}" : "#,
                x1.pretty(fmt),
                x1.t.pretty(fmt),
                x0.pretty(fmt),
            ),
            mlir::OpKind::Tuple(vs) => write!(
                w,
                r#""arc.make_tuple"({vs}) : ({ts}) ->"#,
                vs = vs.all_pretty(", ", fmt),
                ts = vs.map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", ")
            ),
            mlir::OpKind::UnOp(_, _) => crate::todo!(),
            mlir::OpKind::If(v, r0, r1) => write!(
                w,
                r#""arc.if"({v}) ({r0},{r1}) : (i1) ->"#,
                v = v.pretty(fmt),
                r0 = r0.pretty(fmt),
                r1 = r1.pretty(fmt),
            ),
            mlir::OpKind::Emit(_) => crate::todo!(),
            mlir::OpKind::Loop(_) => crate::todo!(),
            mlir::OpKind::Call(v, vs) => write!(
                w,
                r#"call @{callee}({args}) : ({tys}) ->"#,
                callee = v.pretty(fmt),
                args = vs.iter().all_pretty(", ", fmt),
                tys = vs
                    .iter()
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", ")
            ),
            mlir::OpKind::CallIndirect(v, vs) => write!(
                w,
                r#"call_indirect {callee}({args}) : ({tys}) ->"#,
                callee = v.pretty(fmt),
                args = vs.iter().all_pretty(", ", fmt),
                tys = vs
                    .iter()
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
            ),
            mlir::OpKind::CallMethod(v, x, vs) => write!(
                w,
                r#"call_method {object} @{method}({args}) : ({tys}) ->"#,
                object = v.pretty(fmt),
                method = x.pretty(fmt),
                args = vs.iter().all_pretty(", ", fmt),
                tys = vs
                    .iter()
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
            ),
            mlir::OpKind::Return(v)
                if matches!(fmt.types.resolve(v.t),
                    hir::TypeKind::Scalar(hir::ScalarKind::Unit)) =>
                        write!(w, r#"return {s0}"#, s0 = fmt),
            mlir::OpKind::Return(v) => write!(
                w,
                r#"return {v} : {t}{s0}"#,
                v = v.pretty(fmt),
                t = v.t.pretty(fmt),
                s0 = fmt
            ),
            mlir::OpKind::Res(v) => write!(
                w,
                r#""arc.block.result"({v}) : ({t}) -> (){s0}"#,
                v = v.pretty(fmt),
                t = v.t.pretty(fmt),
                s0 = fmt
            ),
            mlir::OpKind::Access(v, i) => write!(
                w,
                r#""arc.struct_access"({v}) {{ field = "{i}" }} : ({t}) ->"#,
                v = v.pretty(fmt),
                i = i.pretty(fmt),
                t = v.t.pretty(fmt)
            ),
            mlir::OpKind::Project(v, i) => write!(
                w,
                r#""arc.index_tuple"({v}) {{ index = {i} }} : ({t}) ->"#,
                v = v.pretty(fmt),
                t = v.t.pretty(fmt),
                i = i,
            ),
            mlir::OpKind::Break(_) => crate::todo!(),
            mlir::OpKind::Continue => crate::todo!(),
            mlir::OpKind::Log(_) => crate::todo!(),
        }
    },
    mlir::Region => write!(w, "{{{}}}", node.blocks.iter().all_pretty("", fmt.indent())),
    mlir::Block => write!(w, "{}",
        node.ops.iter().map_pretty(|op, w| write!(w, "{}{}", fmt.indent(), op.pretty(fmt)), "")),
    mlir::Type => match fmt.types.resolve(*node) {
        mlir::TypeKind::Scalar(kind) => match kind {
            mlir::ScalarKind::I8       => write!(w, "si8"),
            mlir::ScalarKind::I16      => write!(w, "si16"),
            mlir::ScalarKind::I32      => write!(w, "si32"),
            mlir::ScalarKind::I64      => write!(w, "si64"),
            mlir::ScalarKind::U8       => write!(w, "ui8"),
            mlir::ScalarKind::U16      => write!(w, "ui16"),
            mlir::ScalarKind::U32      => write!(w, "ui32"),
            mlir::ScalarKind::U64      => write!(w, "ui64"),
            mlir::ScalarKind::Bf16     => write!(w, "bf16"),
            mlir::ScalarKind::F16      => write!(w, "f16"),
            mlir::ScalarKind::F32      => write!(w, "f32"),
            mlir::ScalarKind::F64      => write!(w, "f64"),
            mlir::ScalarKind::Bool     => write!(w, "i1"),
            mlir::ScalarKind::Unit     => write!(w, "none"),
            mlir::ScalarKind::Size     => write!(w, "ui64"),
            mlir::ScalarKind::DateTime => write!(w, "ui64"),
            mlir::ScalarKind::Duration => write!(w, "ui64"),
            mlir::ScalarKind::Str      => crate::todo!(),
            mlir::ScalarKind::Char     => crate::todo!(),
        }
        mlir::TypeKind::Struct(fs) => write!(w, "!arc.struct<{}>",
            fs.map_pretty(|(x, e), w| write!(w, "{}: {}", x.pretty(fmt), e.pretty(fmt)), ", ")),
        mlir::TypeKind::Nominal(x) => match &fmt.mlir.resolve(&x).kind {
            mlir::ItemKind::Enum(item) => write!(w, "!arc.enum<{}>", item.variants.iter().all_pretty(", ", fmt)),
            mlir::ItemKind::Fun(_)     => unreachable!(),
            mlir::ItemKind::State(_)   => unreachable!(),
            mlir::ItemKind::Task(_)    => unreachable!(),
        }
        mlir::TypeKind::Array(_ty, _sh) => crate::todo!(),
        mlir::TypeKind::Stream(ty)      => write!(w, "arc.stream<{}>", ty.pretty(fmt)),
        mlir::TypeKind::Tuple(tys)      => write!(w, "tuple<{}>", tys.iter().all_pretty(", ", fmt)),
        mlir::TypeKind::Fun(tys, ty)    => write!(w, "({}) -> {}", tys.iter().all_pretty(", ", fmt), ty.pretty(fmt)),
        mlir::TypeKind::Unknown(_)      => unreachable!(),
        mlir::TypeKind::Err             => unreachable!(),
    },
}
