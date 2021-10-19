#![allow(clippy::useless_format)]

#[path = "../pretty.rs"]
#[macro_use]
pub(crate) mod pretty;
use pretty::*;

use crate::hir;
use crate::hir::{Name, Path};

use crate::info::paths::PathId;
use crate::info::Info;
use crate::mlir;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;

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

    mlir::MLIR => write!(w, "module @toplevel {{{items}}}",
        items = node.items
            .iter()
            .filter_map(|i| {
                let x = node.resolve(i);
                  match x.kind {
                    // There is no need to "declare" enum
                    // types before use in MLIR, so just
                    // filter them out
                    mlir::ItemKind::Enum(_) => None,
                    _ => Some(x),
                }})
            .map_pretty(|i, w| write!(w, "{}{}", fmt.indent(), i.pretty(fmt)), ""),
    ),
    mlir::Item => match &node.kind {
        mlir::ItemKind::Fun(item)        => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::Enum(item)       => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::Task(item)       => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::ExternFun(item)  => write!(w, "{}", item.pretty(fmt)),
        mlir::ItemKind::ExternType(item) => write!(w, "{}", item.pretty(fmt)),
    },
    mlir::Fun => write!(w, "func @{id}({params}) -> {t} {body}\n",
        id = node.path.pretty(fmt),
        params = node
            .params
            .iter()
            .filter(|p| matches!(&p.kind, mlir::VarKind::Ok(_)))
            .map_pretty(|p, w| write!(w, "{}", p.pretty(fmt)), ", "),
        t = node.t.pretty(fmt),
        body = node.body.pretty(&fmt.indent()),
    ),
    mlir::ExternFun => write!(w, "func private @{id}({params}) -> {rt}\n",
        id = node.path.pretty(fmt),
        params = node
            .params
            .iter()
            .filter(|p| matches!(&p.kind, mlir::VarKind::Ok(_)))
            .map_pretty(|p, w| write!(w, "{}", p.pretty(fmt)), ", "),
        rt = node.rt.pretty(fmt),
    ),
    mlir::ExternType => write!(w, "func private @{id}({params}) -> !arc.adt<\"{id}\">\n{items}",
        id = node.path.pretty(fmt),
        params = node
            .params
            .iter()
            .filter(|p| matches!(&p.kind, mlir::VarKind::Ok(_)))
            .map_pretty(|p, w| write!(w, "{}", p.pretty(fmt)), ", "),
        items = node.items.iter().map_pretty(|p, w| write!(w, "{}{}", fmt.indent(), fmt.ctx.mlir.resolve(p).pretty(fmt)), ""),
    ),
    mlir::Name => write!(w, "{}", fmt.names.resolve(node)),
    mlir::Task => {
        write!(w,
    r#"func private @{name}({params}) -> (({istream_ts}) -> ({ostream_ts}))"#,
            name = node.path.pretty(fmt),
            params = node.params.iter().all_pretty(", ", fmt),
            istream_ts = node.istream_ts.iter().all_pretty(", ", fmt),
            ostream_ts = node.ostream_ts.iter().all_pretty(", ", fmt),
        )?;
        write!(w,
    r#"{s1}func @{name}_on_event(
        %this : {this_t},
        {ievent},
        %output_stream : !arc.stream<{oevent_t}>
    ) -> ()
    attributes {{
        "arc.mod_name" = "mod_{name}",
        "arc.task_name" = "{name}",
        "arc.is_event_handler"
    }} {body}"#,
            name = node.path.pretty(fmt),
            this_t = node.this_t.pretty(fmt),
            ievent = node.ievent.pretty(fmt),
            oevent_t = node.oevent_t.pretty(fmt),
            body = node.on_event.pretty(fmt.indent()),
            s1 = fmt.indent(),
        )?;
        write!(w,
    r#"{s1}func @{name}_on_start(
        %this : {this_t}
    ) -> ()
    attributes {{
        "arc.mod_name" = "mod_{name}",
        "arc.task_name" = "{name}",
        "arc.is_init"
    }} {body}"#,
            name = node.path.pretty(fmt),
            this_t = node.this_t.pretty(fmt),
            body = node.on_start.pretty(fmt.indent()),
            s1 = fmt.indent(),
        )?;
    },
    mlir::Param => match &node.kind {
        mlir::VarKind::Ok(x) => write!(w, "%{}: {}", x.pretty(fmt), node.t.pretty(fmt)),
        mlir::VarKind::Elided => write!(w, "/* elided */"),
    },
    mlir::Path => write!(w, "{}", node.id.pretty(fmt)),
    mlir::PathId => {
        let kind = fmt.paths.resolve(*node);
        if let Some(id) = kind.pred {
            write!(w, "{}_", id.pretty(fmt))?;
        }
        write!(w, "{}", kind.name.pretty(fmt))
    },
    mlir::Enum => write!(w, "!arc.enum<{}>", node.variants.iter().all_pretty(", ", fmt)),
    mlir::Variant => {
        if node.t.is_unit(fmt.ctx.info) {
            write!(w, "{} : none", node.path.pretty(fmt))
        } else {
            write!(w, "{} : {}", node.path.pretty(fmt), node.t.pretty(fmt))
        }
    },
    mlir::VarKind => match node {
        mlir::VarKind::Ok(x) => write!(w, "%{}", x.pretty(fmt)),
        mlir::VarKind::Elided => write!(w, "/* elided */"),
    },
    mlir::Var => write!(w, "{}", node.kind.pretty(fmt)),
    Vec<mlir::Var> => write!(w, "{}",
        node.iter()
            .filter(|v| matches!(&v.kind, mlir::VarKind::Ok(_)))
            .all_pretty(", ", fmt)
    ),
    mlir::Op => {
        use mlir::ConstKind::*;
        use mlir::OpKind::*;
        use mlir::BinOpKind::*;
        if let mlir::VarKind::Ok(x) = &node.param.kind {
            write!(w, "%{} = ", x.pretty(fmt))?;
        }
        let rt = node.param.t.pretty(fmt);
        match &node.kind {
            mlir::OpKind::Const(c) => match c {
                mlir::ConstKind::Bool(true)  => write!(w, r#"arith.constant true"#),
                mlir::ConstKind::Bool(false) => write!(w, r#"arith.constant false"#),
                mlir::ConstKind::F32(l)      => write!(w, r#"arith.constant {} : {}"#, ryu::Buffer::new().format(*l), rt),
                mlir::ConstKind::F64(l)      => write!(w, r#"arith.constant {} : {}"#, ryu::Buffer::new().format(*l), rt),
                mlir::ConstKind::I8(v)       => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::I16(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::I32(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::I64(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::U8(v)       => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::U16(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::U32(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::U64(v)      => write!(w, r#"arc.constant {} : {}"#, v, rt),
                mlir::ConstKind::Fun(x)      => write!(w, r#"constant @{} : {}"#, x.pretty(fmt), rt),
                mlir::ConstKind::Char(_)     => crate::todo!(),
                mlir::ConstKind::Time(_)     => crate::todo!(),
                mlir::ConstKind::Noop        => write!(w, r#"// noop"#),
            },
            mlir::OpKind::BinOp(l, op, r) => {
                let t = l.t;
                let l = l.pretty(fmt);
                let r = r.pretty(fmt);
                let info = fmt.info;
                let lt = t.pretty(fmt);
                use hir::ScalarKind::*;
                use hir::TypeKind::*;
                match &op.kind {
                    Add  if t.is_int(info)   => write!(w, r#"arc.addi {}, {} : {}"#, l, r, lt),
                    Add  if t.is_float(info) => write!(w, r#"arith.addf {}, {} : {}"#, l, r, lt),
                    Sub  if t.is_int(info)   => write!(w, r#"arc.subi {}, {} : {}"#, l, r, lt),
                    Sub  if t.is_float(info) => write!(w, r#"arith.subf {}, {} : {}"#, l, r, lt),
                    Mul  if t.is_int(info)   => write!(w, r#"arc.muli {}, {} : {}"#, l, r, lt),
                    Mul  if t.is_float(info) => write!(w, r#"arith.mulf {}, {} : {}"#, l, r, lt),
                    Div  if t.is_int(info)   => write!(w, r#"arc.divi {}, {} : {}"#, l, r, lt),
                    Div  if t.is_float(info) => write!(w, r#"arith.divf {}, {} : {}"#, l, r, lt),
                    Mod  if t.is_int(info)   => write!(w, r#"arc.remi {}, {} : {}"#, l, r, lt),
                    Mod  if t.is_float(info) => write!(w, r#"arith.remf {}, {} : {}"#, l, r, lt),
                    Pow  if t.is_int(info)   => write!(w, r#"arc.powi {}, {} : {}"#, l, r, lt),
                    Pow  if t.is_float(info) => write!(w, r#"math.powf {}, {} : {}"#, l, r, lt),
                    Lt   if t.is_int(info)   => write!(w, r#"arc.cmpi lt, {}, {} : {}"#, l, r, lt),
                    Lt   if t.is_float(info) => write!(w, r#"arith.cmpf olt, {}, {} : {}"#, l, r, lt),
                    Leq  if t.is_int(info)   => write!(w, r#"arc.cmpi le, {}, {} : {}"#, l, r, lt),
                    Leq  if t.is_float(info) => write!(w, r#"arith.cmpf ole, {}, {} : {}"#, l, r, lt),
                    Gt   if t.is_int(info)   => write!(w, r#"arc.cmpi gt, {}, {} : {}"#, l, r, lt),
                    Gt   if t.is_float(info) => write!(w, r#"arith.cmpf ogt, {}, {} : {}"#, l, r, lt),
                    Geq  if t.is_int(info)   => write!(w, r#"arc.cmpi ge, {}, {} : {}"#, l, r, lt),
                    Geq  if t.is_float(info) => write!(w, r#"arith.cmpf oge, {}, {} : {}"#, l, r, lt),
                    Equ  if t.is_int(info)   => write!(w, r#"arc.cmpi eq, {}, {} : {}"#, l, r, lt),
                    Equ  if t.is_float(info) => write!(w, r#"arith.cmpf oeq, {}, {} : {}"#, l, r, lt),
                    Equ  if t.is_bool(info)  => write!(w, r#"arith.cmpi eq, {}, {} : {}"#, l, r, lt),
                    Neq  if t.is_int(info)   => write!(w, r#"arc.cmpi ne, {}, {} : {}"#, l, r, lt),
                    Neq  if t.is_float(info) => write!(w, r#"arith.cmpf one, {}, {} : {}"#, l, r, lt),
                    Neq  if t.is_bool(info)  => write!(w, r#"arith.cmpi ne, {}, {} : {}"#, l, r, lt),
                    And  if t.is_bool(info)  => write!(w, r#"arith.andi {}, {} : {}"#, l, r, lt),
                    Or   if t.is_bool(info)  => write!(w, r#"arith.ori {}, {} : {}"#, l, r, lt),
                    Xor  if t.is_bool(info)  => write!(w, r#"arith.xori {}, {} : {}"#, l, r, lt),
                    Band if t.is_int(info)   => write!(w, r#"arc.and {}, {} : {}"#, l, r, lt),
                    Bor  if t.is_int(info)   => write!(w, r#"arc.or {}, {} : {}"#, l, r, lt),
                    Bxor if t.is_int(info)   => write!(w, r#"arc.xor {}, {} : {}"#, l, r, lt),
                    op => {
                        let kind = info.types.resolve(*t);
                        unreachable!("Undefined op: {:?} for {:?}", op, kind)
                    },
                }
            }
            mlir::OpKind::Array(_) => todo!(),
            mlir::OpKind::Struct(xfs) => write!(
                w,
                r#"arc.make_struct({xfs} : {ts}) : {rt}"#,
                xfs = xfs.values().all_pretty(", ", fmt),
                ts = xfs
                    .values()
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
                rt = rt,
            ),
              // %s = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32>
            mlir::OpKind::Enwrap(x, v) => if v.t.is_unit(fmt.ctx.info) {
                write!(w, r#"arc.make_enum () as "{}" : {}"#, x.pretty(fmt), rt)
            } else {
                write!(w, r#"arc.make_enum ({} : {}) as "{}" : {}"#, v.pretty(fmt), v.t.pretty(fmt), x.pretty(fmt), rt)
            },
              // %r = arc.enum_access "b" in (%e : !arc.enum<a : i32, b : f32>) : f32
            mlir::OpKind::Unwrap(x, v) => if node.param.t.is_unit(fmt.ctx.info) {
                write!(w, r#"arc.enum_access "{}" in ({} : {}) : none"#, x.pretty(fmt), v.pretty(fmt), v.t.pretty(fmt))
            } else {
                write!(w, r#"arc.enum_access "{}" in ({} : {}) : {}"#, x.pretty(fmt), v.pretty(fmt), v.t.pretty(fmt), rt)
            },
                // %r = arc.enum_check (%e : !arc.enum<a : i32, b : f32>) is "a" : i1
            mlir::OpKind::Is(x, v) => write!(
                w,
                r#"arc.enum_check ({} : {}) is "{}" : {}"#,
                v.pretty(fmt),
                v.t.pretty(fmt),
                x.pretty(fmt),
                rt,
            ),
            mlir::OpKind::Tuple(vs) => write!(
                w,
                r#""arc.make_tuple"({vs}) : ({ts}) -> {rt}"#,
                vs = vs.all_pretty(", ", fmt),
                ts = vs.map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
                rt = SpecialCase::new(node.param.t).pretty(fmt),
            ),
            mlir::OpKind::UnOp(_, _) => crate::todo!(),
            mlir::OpKind::If(v, b0, b1) => write!(
                w,
                r#""arc.if"({v}) ({b0},{b1}) : (i1) -> {rt}"#,
                v = v.pretty(fmt),
                b0 = b0.pretty(fmt.indent()),
                b1 = b1.pretty(fmt.indent()),
                rt = rt,
            ),
            mlir::OpKind::Emit(v) => write!(
                w,
                r#""arc.emit"({v}, %output_stream) : ({t}, !arc.stream<{t}>) -> {rt}"#,
                v = v.pretty(fmt),
                t = v.t.pretty(fmt),
                rt = rt
            ),
            mlir::OpKind::Loop(_) => crate::todo!(),
            mlir::OpKind::Call(v, vs) => write!(
                w,
                r#"call @{callee}({args}) : ({ts}) -> {rt}"#,
                callee = v.pretty(fmt),
                args = vs.pretty(fmt),
                ts = vs
                    .iter()
                    .filter(|v| !v.t.is_unit(fmt.ctx.info))
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
                rt = SpecialCase::new(node.param.t).pretty(fmt),
            ),
            mlir::OpKind::CallIndirect(v, vs) => write!(
                w,
                r#"call_indirect {callee}({args}) : ({ts}) -> {rt}"#,
                callee = v.pretty(fmt),
                args = vs.pretty(fmt),
                ts = vs
                    .iter()
                    .filter(|v| !v.t.is_unit(fmt.ctx.info))
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
                rt = SpecialCase::new(node.param.t).pretty(fmt),
            ),
            mlir::OpKind::CallMethod(v, x, vs) => write!(
                w,
                r#"call_method {object} @{method}({args}) : ({ts}) -> {rt}"#,
                object = v.pretty(fmt),
                method = x.pretty(fmt),
                args = vs.pretty(fmt),
                ts = vs
                    .iter()
                    .filter(|v| !v.t.is_unit(fmt.ctx.info))
                    .map_pretty(|v, w| write!(w, "{}", v.t.pretty(fmt)), ", "),
                rt = SpecialCase::new(node.param.t).pretty(fmt),
            ),
            mlir::OpKind::Return(v) => {
                if v.t.is_unit(fmt.ctx.info) {
                          write!(w, "return")
                } else {
                         write!(w, r#"return {} : {}"#, v.pretty(fmt), rt)
                    }
            }
            mlir::OpKind::Result(v) => {
                if v.t.is_unit(fmt.ctx.info) {
                          write!(w, r#""arc.block.result"() : () -> ()"#)
                } else {
                          write!(w, r#""arc.block.result"({v}) : ({t}) -> ()"#,
                              v = v.pretty(fmt),
                              t = v.t.pretty(fmt),
                          )
                    }
            },
            mlir::OpKind::Access(v, i) => write!(
                w,
                r#""arc.struct_access"({v}) {{ field = "{i}" }} : ({t}) -> {rt}"#,
                v = v.pretty(fmt),
                i = i.pretty(fmt),
                t = v.t.pretty(fmt),
                rt = rt,
            ),
            mlir::OpKind::Project(v, i) => write!(
                w,
                r#""arc.index_tuple"({v}) {{ index = {i} }} : ({t}) -> {rt}"#,
                v = v.pretty(fmt),
                t = v.t.pretty(fmt),
                i = i,
                rt = rt,
            ),
            mlir::OpKind::Break(_) => crate::todo!(),
            mlir::OpKind::Continue => crate::todo!(),
            mlir::OpKind::Log(_) => crate::todo!(),
            mlir::OpKind::Noop => write!(w, "// noop"),
            mlir::OpKind::Panic => write!(w, r#""arc.panic"() : () -> ()"#),
        }
    },
    mlir::Block => write!(w, "{{{}{s0}}}",
        node.ops.iter().map_pretty(|op, w| write!(w, "{}{}", fmt.indent(), op.pretty(fmt)), ""),
        s0 = fmt,
    ),
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
            mlir::ScalarKind::F32      => write!(w, "f32"),
            mlir::ScalarKind::F64      => write!(w, "f64"),
            mlir::ScalarKind::Bool     => write!(w, "i1"),
            mlir::ScalarKind::Unit     => write!(w, "()"), // Unit becomes Void
            mlir::ScalarKind::Size     => write!(w, "ui64"),
            mlir::ScalarKind::DateTime => write!(w, "ui64"),
            mlir::ScalarKind::Duration => write!(w, "ui64"),
            mlir::ScalarKind::Str      => write!(w, "!arc.adt<\"String\">"),
            mlir::ScalarKind::Char     => write!(w, "!arc.adt<\"char\">"),
        }
        mlir::TypeKind::Struct(fs) => write!(w, "!arc.struct<{}>",
            fs.map_pretty(|(x, e), w| write!(w, "{}: {}", x.pretty(fmt), e.pretty(fmt)), ", ")),
        mlir::TypeKind::Nominal(x) => match &fmt.mlir.resolve(&x).kind {
            mlir::ItemKind::Enum(item)    => write!(w, "!arc.enum<{}>", item.variants.iter().all_pretty(", ", fmt)),
            mlir::ItemKind::Fun(_)        => unreachable!(),
            mlir::ItemKind::Task(_)       => unreachable!(),
            mlir::ItemKind::ExternType(_) => write!(w, "!arc.adt<\"{}\">", x.pretty(fmt)),
            mlir::ItemKind::ExternFun(_)  => unreachable!(),
        }
        mlir::TypeKind::Array(_t, _sh) => crate::todo!(),
        mlir::TypeKind::Stream(t)      => write!(w, "!arc.stream<{}>", t.pretty(fmt)),
        mlir::TypeKind::Tuple(ts)      => write!(w, "tuple<{}>", ts.iter().all_pretty(", ", fmt)),
        mlir::TypeKind::Fun(ts, t)     => write!(w, "({}) -> {}",
            ts.iter().filter(|t| !t.is_unit(fmt.ctx.info)).all_pretty(", ", fmt),
            SpecialCase::new(t).pretty(fmt),
        ),
        mlir::TypeKind::Unknown(_)     => unreachable!(),
        mlir::TypeKind::Err            => unreachable!(),
    },
    SpecialCase<mlir::Type> => {
        if node.inner.is_fun(fmt.ctx.info) {
            write!(w, "({})", node.inner.pretty(fmt))
        } else {
            write!(w, "{}", node.inner.pretty(fmt))
        }
    },
}

#[derive(New)]
struct SpecialCase<T> {
    inner: T,
}
