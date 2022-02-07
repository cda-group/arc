#![allow(clippy::useless_format)]

#[path = "../pretty.rs"]
#[macro_use]
mod pretty;

use pretty::*;

use crate::hir;
use crate::hir::HIR;
use crate::info::modes::Verbosity;
use crate::info::names::NameId;
use crate::info::paths::PathId;
use crate::info::types::TypeId;
use crate::info::Info;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;
use arc_script_compiler_shared::VecMap;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

#[derive(New, Copy, Clone, Shrinkwrap)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i Info,
    pub(crate) hir: &'i HIR,
}

impl hir::HIR {
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

pretty! {
    [node, fmt, w]

    hir::HIR => write!(w, "{}", node.namespace.iter().map(|x| node.resolve(x)).all_pretty("\n", fmt)),
    hir::Item => write!(w, "{}", node.kind.pretty(fmt)),
    hir::ItemKind => match node {
        hir::ItemKind::Fun(item)        => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::TypeAlias(item)  => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::Enum(item)       => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::Task(item)       => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::ExternFun(item)  => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::ExternType(item) => write!(w, "{}", item.pretty(fmt)),
        hir::ItemKind::Variant(item)    => write!(w, "{}", item.pretty(fmt)),
    },
    hir::Assign => write!(w, "{kind} {param} = {expr}",
        kind = node.kind.pretty(fmt),
        param = node.param.pretty(fmt),
        expr = node.expr.pretty(fmt)
    ),
    hir::MutKind => match node {
        hir::MutKind::Immutable => write!(w, "val"),
        hir::MutKind::Mutable => write!(w, "var"),
    },
    hir::Fun => write!(w, "fun {name}({params}): {rty} {body}",
        name = fmt.paths.resolve(node.path).name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        body = node.body.pretty(fmt),
        rty = node.rt.pretty(fmt),
    ),
    hir::ExternFun => write!(w, "extern fun {name}({params}): {ty};",
        name = fmt.paths.resolve(node.path).name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        ty = node.rt.pretty(fmt),
    ),
    hir::ExternType => write!(w, "extern type {name}({params}) {{{items}{s0}}}",
        name = fmt.paths.resolve(node.path).name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        items = node.items.iter().map_pretty(|f, w| write!(w, "{}{}", fmt.indent(), fmt.ctx.hir.resolve(f).pretty(fmt)), ", "),
        s0 = fmt,
    ),
    hir::Task => write!(w, "task {name}({params}): {iexterior} -> {oexterior} \
        {{\
            {s1}{iinterior}\
            {s1}{ointerior}\
            {fields}\
            {items}\
            {s1}{on_start}\
            {s1}{on_event}\
            {s0}\
        }}",
        name = fmt.paths.resolve(node.path).name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        iexterior = node.iinterface.exterior.iter().all_pretty(", ", fmt),
        oexterior = node.ointerface.exterior.iter().all_pretty(", ", fmt),
        iinterior = fmt.hir.resolve(&node.iinterface.interior).pretty(fmt.indent()),
        ointerior = fmt.hir.resolve(&node.ointerface.interior).pretty(fmt.indent()),
        items = node.namespace.iter().map_pretty(
            |x, w| write!(w, "{s0}{}", fmt.hir.resolve(x).pretty(fmt.indent()), s0 = fmt.indent()),
            ""
        ),
        fields = node.fields.iter().map_pretty(|(x, t), w| 
            write!(w, "{s1}val {}: {} = uninitialised;", x.pretty(fmt), t.pretty(fmt), s1 = fmt.indent()), ""
        ),
        on_start = node.on_start.pretty(fmt.indent()),
        on_event = node.on_event.pretty(fmt.indent()),
        s0 = fmt,
        s1 = fmt.indent(),
    ),
    hir::Interface => { },
    Vec<hir::Param> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    hir::Param => write!(w, "{}: {}", node.kind.pretty(fmt), node.t.pretty(fmt)),
    hir::ParamKind => match node {
        hir::ParamKind::Ok(x)  => write!(w, "{}", x.pretty(fmt))?,
        hir::ParamKind::Ignore => write!(w, "_")?,
        hir::ParamKind::Err    => write!(w, "☇")?,
    },
    hir::Name => write!(w, "{}", node.id.pretty(fmt)),
    hir::NameId => write!(w, "{}", fmt.names.resolve(node)),
    hir::TypeAlias => write!(w, "type {id} = {ty}",
        id = fmt.paths.resolve(node.path).name.pretty(fmt),
        ty = node.t.pretty(fmt)
    ),
    hir::OnStart => write!(w, "{}{s0}{}();",
        fmt.hir.resolve(node.fun).pretty(fmt),
        node.fun.pretty(fmt),
        s0 = fmt,
    ),
    hir::OnEvent => write!(w, "{}{s0}on event => {}(event)",
        fmt.hir.resolve(node.fun).pretty(fmt),
        node.fun.pretty(fmt),
        s0 = fmt,
    )?,
    hir::Path => write!(w, "{}", node.id.pretty(fmt)),
    hir::PathId => {
        let kind = fmt.paths.resolve(node);
        if let Some(id) = kind.pred {
            write!(w, "{}::", id.pretty(fmt))?;
        }
        write!(w, "{}", kind.name.pretty(fmt))
    },
    hir::Enum => {
        let name = fmt.paths.resolve(node.path).name;
        write!(w, "enum {id} {{{variants}{s0}}}",
            id = name.pretty(fmt),
            variants = node.variants.iter().map_pretty(
                |v, w| write!(w, "{s1}{v}",
                    v = fmt.hir.resolve(v).pretty(fmt),
                    s1 = fmt.indent()
                ),
                ","
            ),
            s0 = fmt
        )
    },
    hir::Variant => {
        let kind = fmt.types.resolve(node.t);
        let name = fmt.paths.resolve(node.path).name;
        if let hir::TypeKind::Scalar(hir::ScalarKind::Unit) = kind {
            write!(w, "{}", name.pretty(fmt))
        } else {
            write!(w, "{}({})", name.pretty(fmt), node.t.pretty(fmt))
        }
    },
    hir::Stmt => match &node.kind {
        hir::StmtKind::Assign(item) => write!(w, "{};", item.pretty(fmt)),
    },
    VecDeque<hir::Stmt> => write!(w, "{}", node.iter().map_pretty(|stmt, w| write!(w, "{}{}", stmt.pretty(fmt), fmt), "")),
    hir::Block => write!(w, "{{{s1}{stmts}{var}{s0}}}",
        stmts = node.stmts.pretty(fmt.indent()),
        var = node.var.pretty(fmt.indent()),
        s0 = fmt,
        s1 = fmt.indent()
    ),
    hir::Expr => {
        let kind = fmt.hir.exprs.resolve(node);
        if fmt.mode.verbosity >= Verbosity::Debug {
            write!(w, "({}):{}", kind.pretty(fmt), node.t.pretty(fmt))?;
        } else {
            write!(w, "{}", kind.pretty(fmt))?;
        }
    },
    hir::Var => {
        if fmt.mode.verbosity >= Verbosity::Debug {
            write!(w, "{}:{}", node.kind.pretty(fmt), node.t.pretty(fmt))
        } else {
            write!(w, "{}", node.kind.pretty(fmt))
        }
    },
    hir::VarKind => {
        match node {
            hir::VarKind::Ok(x, _) => write!(w, "{}", x.pretty(fmt)),
            hir::VarKind::Err => write!(w, "☇"),
        }
    },
    hir::UnOp => write!(w, "{}", node.kind.pretty(fmt)),
    hir::UnOpKind => match node {
        hir::UnOpKind::Not => write!(w, "not "),
        hir::UnOpKind::Neg => write!(w, "-"),
        hir::UnOpKind::Err => write!(w, "☇"),
        _ => unreachable!()
    },
    Vec<hir::Var> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    hir::ExprKind => match node {
        hir::ExprKind::Return(v)         => write!(w, "return {}", v.pretty(fmt)),
        hir::ExprKind::Break(v)          => write!(w, "break {}", v.pretty(fmt)),
        hir::ExprKind::Continue          => write!(w, "continue"),
        hir::ExprKind::After(v0, v1)     => write!(w, "after {} {}", v0.pretty(fmt), v1.pretty(fmt)),
        hir::ExprKind::Every(v0, v1)     => write!(w, "every {} {}", v0.pretty(fmt), v1.pretty(fmt)),
        hir::ExprKind::Cast(v, ty)       => write!(w, "{} as {}", v.pretty(fmt), ty.pretty(fmt)),
        hir::ExprKind::If(v, b0, b1)     => write!(w, "if {} {} else {}", v.pretty(fmt), b0.pretty(fmt), b1.pretty(fmt)),
        hir::ExprKind::Lit(l)            => write!(w, "{}", l.pretty(fmt)),
        hir::ExprKind::BinOp(v0, op, v1) => write!(w, "{}{}{}", v0.pretty(fmt), op.pretty(fmt), v1.pretty(fmt)),
        hir::ExprKind::UnOp(op, v)       => write!(w, "{}{}", op.pretty(fmt), v.pretty(fmt)),
        hir::ExprKind::Project(v, i)     => write!(w, "{}.{}", v.pretty(fmt), i.id),
        hir::ExprKind::Access(v, x)      => write!(w, "{}.{}", v.pretty(fmt), x.pretty(fmt)),
        hir::ExprKind::Call(v, vs)       => write!(w, "{}({})", v.pretty(fmt), vs.pretty(fmt)),
        hir::ExprKind::SelfCall(x, vs)   => write!(w, "{}({})", x.pretty(fmt), vs.pretty(fmt)),
        hir::ExprKind::Invoke(v, x, vs)  => write!(w, "{}.{}({})", v.pretty(fmt), x.pretty(fmt), vs.pretty(fmt)),
        hir::ExprKind::Select(v, vs)     => write!(w, "{}[{}]", v.pretty(fmt), vs.pretty(fmt)),
        hir::ExprKind::Emit(v)           => write!(w, "emit {}", v.pretty(fmt)),
        hir::ExprKind::Log(v)            => write!(w, "log {}", v.pretty(fmt)),
        hir::ExprKind::Array(vs)         => write!(w, "[{}]", vs.pretty(fmt)),
        hir::ExprKind::Struct(fs)        => write!(w, "{{ {} }}", fs.pretty(fmt)),
        hir::ExprKind::Tuple(vs)         => write!(w, "({})", vs.pretty(fmt)),
        hir::ExprKind::Loop(v)           => write!(w, "loop {{{s1}{}}}", v.pretty(fmt), s1 = fmt.indent()),
        hir::ExprKind::Unwrap(x, v)      => write!(w, "unwrap[{}]({})", x.pretty(fmt), v.pretty(fmt)),
        hir::ExprKind::Enwrap(x, v)      => write!(w, "enwrap[{}]({})", x.pretty(fmt), v.pretty(fmt)),
        hir::ExprKind::Is(x, v)          => write!(w, "is[{}]({})", v.pretty(fmt), v.pretty(fmt)),
        hir::ExprKind::Item(x)           => write!(w, "{}", x.pretty(fmt)),
        hir::ExprKind::Unreachable       => write!(w, "unreachable"),
        hir::ExprKind::Initialise(x, v)  => write!(w, "initialise[{}]({})", x.pretty(fmt), v.pretty(fmt)),
        hir::ExprKind::Err               => write!(w, "☇"),
    },
    hir::LitKind => match node {
        hir::LitKind::I8(l)       => write!(w, "{}i8", l),
        hir::LitKind::I16(l)      => write!(w, "{}i16", l),
        hir::LitKind::I32(l)      => write!(w, "{}", l),
        hir::LitKind::I64(l)      => write!(w, "{}i64", l),
        hir::LitKind::U8(l)       => write!(w, "{}u8", l),
        hir::LitKind::U16(l)      => write!(w, "{}u16", l),
        hir::LitKind::U32(l)      => write!(w, "{}u32", l),
        hir::LitKind::U64(l)      => write!(w, "{}u64", l),
        hir::LitKind::F32(l)      => write!(w, "{}f32", ryu::Buffer::new().format(*l)),
        hir::LitKind::F64(l)      => write!(w, "{}", ryu::Buffer::new().format(*l)),
        hir::LitKind::Bool(l)     => write!(w, "{}", l),
        hir::LitKind::Char(l)     => write!(w, "'{}'", l),
        hir::LitKind::Str(l)      => write!(w, r#""{}""#, l),
        hir::LitKind::DateTime(l) => write!(w, "{}", l),
        hir::LitKind::Duration(l) => write!(w, "{}", l.as_seconds_f64()),
        hir::LitKind::Unit        => write!(w, "unit"),
        hir::LitKind::Err         => write!(w, "☇"),
    },
    hir::BinOp => match &node.kind {
        hir::BinOpKind::Add  => write!(w, " + "),
        hir::BinOpKind::Sub  => write!(w, " - "),
        hir::BinOpKind::Mul  => write!(w, " * "),
        hir::BinOpKind::Div  => write!(w, " / "),
        hir::BinOpKind::Mod  => write!(w, " % "),
        hir::BinOpKind::Pow  => write!(w, " ** "),
        hir::BinOpKind::Equ  => write!(w, " == "),
        hir::BinOpKind::Neq  => write!(w, " != "),
        hir::BinOpKind::Gt   => write!(w, " > "),
        hir::BinOpKind::In   => write!(w, " in "),
        hir::BinOpKind::Lt   => write!(w, " < "),
        hir::BinOpKind::Geq  => write!(w, " >= "),
        hir::BinOpKind::Leq  => write!(w, " <= "),
        hir::BinOpKind::Or   => write!(w, " or "),
        hir::BinOpKind::And  => write!(w, " and "),
        hir::BinOpKind::Xor  => write!(w, " xor "),
        hir::BinOpKind::Band => write!(w, " band "),
        hir::BinOpKind::Bor  => write!(w, " bor "),
        hir::BinOpKind::Bxor => write!(w, " bxor "),
        hir::BinOpKind::Mut  => write!(w, " = "),
        hir::BinOpKind::Err  => write!(w, " ☇ "),
    },
    hir::ScalarKind => match node {
        hir::ScalarKind::Bool     => write!(w, "bool"),
        hir::ScalarKind::Char     => write!(w, "char"),
        hir::ScalarKind::F32      => write!(w, "f32"),
        hir::ScalarKind::F64      => write!(w, "f64"),
        hir::ScalarKind::I8       => write!(w, "i8"),
        hir::ScalarKind::I16      => write!(w, "i16"),
        hir::ScalarKind::I32      => write!(w, "i32"),
        hir::ScalarKind::I64      => write!(w, "i64"),
        hir::ScalarKind::U8       => write!(w, "u8"),
        hir::ScalarKind::U16      => write!(w, "u16"),
        hir::ScalarKind::U32      => write!(w, "u32"),
        hir::ScalarKind::U64      => write!(w, "u64"),
        hir::ScalarKind::Str      => write!(w, "str"),
        hir::ScalarKind::Unit     => write!(w, "unit"),
        hir::ScalarKind::Size     => write!(w, "size"),
        hir::ScalarKind::DateTime => write!(w, "time"),
        hir::ScalarKind::Duration => write!(w, "duration"),
    },
    hir::Type => match fmt.types.resolve(node) {
        hir::TypeKind::Scalar(kind) => write!(w, "{}", kind.pretty(fmt)),
        hir::TypeKind::Struct(fs) => {
            write!(w, "{{ {} }}",
                fs.map_pretty(|(x, t), w| write!(w, "{}: {}", x.pretty(fmt), t.pretty(fmt)), ", "))
        }
        hir::TypeKind::Nominal(x)    => write!(w, "{}", x.pretty(fmt)),
        hir::TypeKind::Array(ty, sh) => write!(w, "[{}; {}]", ty.pretty(fmt), sh.pretty(fmt)),
        hir::TypeKind::Stream(ty)    => write!(w, "~{}", ty.pretty(fmt)),
        hir::TypeKind::Tuple(tys)    => write!(w, "({})", tys.iter().all_pretty(", ", fmt)),
        hir::TypeKind::Fun(args, ty) => write!(w, "fun({}): {}", args.iter().all_pretty(", ", fmt), ty.pretty(fmt)),
        hir::TypeKind::Unknown(_)    => write!(w, "'{}", node.id.0),
        hir::TypeKind::Err           => write!(w, "☇"),
    },
    hir::Shape => write!(w, "{}", node.dims.iter().all_pretty(", ", fmt)),
    hir::Dim => match &node.kind {
        hir::DimKind::Var(_)       => write!(w, "?"),
        hir::DimKind::Val(v)       => write!(w, "{}", v),
        hir::DimKind::Op(l, op, r) => write!(w, "{}{}{}", l.pretty(fmt), op.pretty(fmt), r.pretty(fmt)),
        hir::DimKind::Err          => write!(w, "☇"),
    },
    hir::DimOp => match &node.kind {
        hir::DimOpKind::Add => write!(w, "+"),
        hir::DimOpKind::Sub => write!(w, "-"),
        hir::DimOpKind::Mul => write!(w, "*"),
        hir::DimOpKind::Div => write!(w, "/"),
    },
    Box<hir::Dim> => write!(w, "{}", node.as_ref().pretty(fmt)),
    Vec<hir::Expr> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    Vec<hir::Type> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    VecMap<hir::Name, hir::Var> => {
        write!(w, "{}", node.map_pretty(|(x, v), w| write!(w, "{}: {}", x.pretty(fmt), v.pretty(fmt)), ", "))
    },
}
