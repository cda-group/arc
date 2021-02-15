#![allow(clippy::useless_format)]
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::pretty::*;
use arc_script_core_shared::New;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

#[derive(New, Copy, Clone)]
pub(crate) struct Context<'i> {
    info: &'i Info,
    hir: &'i HIR,
}

pub(crate) fn pretty<'i, 'j, Node>(
    node: &'i Node,
    hir: &'j HIR,
    info: &'j Info,
) -> Pretty<'i, Node, Context<'j>> {
    node.to_pretty(Context::new(info, hir))
}

impl<'i> Display for Pretty<'i, HIR, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(hir, fmt) = self;
        write!(
            f,
            "{}",
            hir.items
                .iter()
                .filter_map(|x| hir.defs.get(x))
                .all_pretty("\n", fmt)
        );
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, hir::Item, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match &item.kind {
            hir::ItemKind::Fun(item)     => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::Alias(item)   => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::Enum(item)    => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::Task(item)    => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::State(item)   => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::Extern(item)  => write!(f, "{}", item.pretty(fmt)),
            hir::ItemKind::Variant(item) => write!(f, "{}", item.pretty(fmt)),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Fun, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "fun {id}({params}) -> {ty} {{{s1}{body}{s0}}}",
            id = name.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            ty = item.rtv.pretty(fmt),
            body = item.body.pretty(&fmt.indent()),
            s0 = fmt,
            s1 = fmt.indent(),
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Extern, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "extern fun {id}({params}) -> {ty};",
            id = name.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            ty = item.rtv.pretty(fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Param, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(p, fmt) = self;
        match p.kind {
            hir::ParamKind::Var(x) => write!(f, "{}", x.pretty(fmt))?,
            hir::ParamKind::Ignore => write!(f, "_")?,
            hir::ParamKind::Err => write!(f, "☇")?,
        }
        write!(f, ": {}", p.tv.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, hir::Name, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(x, fmt) = self;
        write!(f, "{}", x.id.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, NameId, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(id, fmt) = self;
        write!(f, "{}", fmt.ctx.info.names.resolve(**id))
    }
}

impl<'i> Display for Pretty<'i, hir::State, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "state {name}: {ty} = {init};",
            name = name.pretty(fmt),
            ty = item.tv.pretty(fmt),
            init = item.init.pretty(fmt)
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Alias, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "type {id} = {ty}",
            id = name.pretty(fmt),
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

impl<'i> Display for Pretty<'i, hir::Task, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "task {name}({params}) {ihub} -> {ohub} {{{items}{s1}{on}{s0}}}",
            name = name.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            ihub = item.ihub.pretty(fmt),
            ohub = item.ohub.pretty(fmt),
            items = item.items.iter().map_pretty(
                |x, f| write!(
                    f,
                    "{s0}{}",
                    fmt.ctx.hir.defs.get(x).unwrap().pretty(fmt.indent()),
                    s0 = fmt.indent()
                ),
                ""
            ),
            on = item.on.pretty(fmt.indent()),
            s0 = fmt,
            s1 = fmt.indent(),
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Hub, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match item.kind {
            hir::HubKind::Tagged(x) => {
                let item = fmt.ctx.hir.defs.get(&x).unwrap();
                if let hir::ItemKind::Enum(item) = &item.kind {
                    write!(
                        f,
                        "({})",
                        item.variants.iter().map_pretty(
                            |v, f| write!(f, "{}", fmt.ctx.hir.defs.get(v).unwrap().pretty(fmt)),
                            ", "
                        )
                    )
                } else {
                    unreachable!()
                }
            }
            hir::HubKind::Single(tv) => write!(f, "({})", tv.pretty(fmt)),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::On, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(on, fmt) = self;
        write!(
            f,
            "on {{{s1}{param} => {{{s2}{body}{s1}}}{s0}}}",
            param = on.param.pretty(fmt),
            body = on.body.pretty(fmt.indent().indent()),
            s0 = fmt,
            s1 = fmt.indent(),
            s2 = fmt.indent().indent()
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Path, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, fmt) = self;
        write!(f, "{}", path.id.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, PathId, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, fmt) = self;
        let path = fmt.ctx.info.paths.resolve(*path);
        if let Some(id) = path.pred {
            write!(f, "{}::", id.pretty(fmt))?;
        }
        write!(f, "{}", path.name.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, hir::Enum, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        write!(
            f,
            "enum {id} {{{variants}{s0}}}",
            id = name.pretty(fmt),
            variants = item.variants.iter().map_pretty(
                |v, f| write!(
                    f,
                    "{s1}{v}",
                    v = fmt.ctx.hir.defs.get(v).unwrap().pretty(fmt),
                    s1 = fmt.indent()
                ),
                ","
            ),
            s0 = fmt
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Variant, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        let ty = fmt.ctx.info.types.resolve(item.tv);
        let name = fmt.ctx.info.paths.resolve(item.path.id).name;
        if let hir::TypeKind::Scalar(hir::ScalarKind::Unit) = ty.kind {
            write!(f, "{}", name.pretty(fmt),)
        } else {
            write!(f, "{}({})", name.pretty(fmt), ty.pretty(fmt))
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Expr, Context<'_>> {
    #[allow(clippy::many_single_char_names)]
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(expr, fmt) = self;
        match &expr.kind {
            hir::ExprKind::If(e0, e1, e2) => write!(
                f,
                "if {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                e0 = e0.pretty(&fmt),
                e1 = e1.pretty(&fmt.indent()),
                e2 = e2.pretty(&fmt.indent()),
                s0 = fmt,
                s1 = fmt.indent(),
            ),
            hir::ExprKind::Let(p, e0, e1) => write!(
                f,
                "let {p} = {e0} in{s}{e1}",
                p = p.pretty(fmt),
                e0 = e0.pretty(fmt),
                e1 = e1.pretty(fmt),
                s = fmt
            ),
            hir::ExprKind::Lit(l) => write!(f, "{}", l.pretty(fmt)),
            hir::ExprKind::BinOp(e0, op, e1) => write!(
                f,
                "{e0}{op}{e1}",
                e0 = e0.pretty(fmt),
                op = op.pretty(fmt),
                e1 = e1.pretty(fmt)
            ),
            hir::ExprKind::UnOp(op, e0) => match &op.kind {
                hir::UnOpKind::Not => write!(f, "not {}", e0.pretty(fmt)),
                hir::UnOpKind::Neg => write!(f, "-{}", e0.pretty(fmt)),
                hir::UnOpKind::Err => write!(f, "☇{}", e0.pretty(fmt)),
            },
            hir::ExprKind::Project(e, i) => write!(f, "{}.{}", e.pretty(fmt), i.id),
            hir::ExprKind::Access(e, x) => write!(f, "{}.{}", e.pretty(fmt), x.pretty(fmt)),
            hir::ExprKind::Call(e, es) => write!(f, "{}({})", e.pretty(fmt), es.iter().all_pretty(", ", fmt)),
            hir::ExprKind::Emit(e) => write!(f, "emit {e}", e = e.pretty(fmt)),
            hir::ExprKind::Log(e) => write!(f, "log {e}", e = e.pretty(fmt)),
            hir::ExprKind::Array(es) => write!(f, "[{es}]", es = es.all_pretty(", ", fmt)),
            hir::ExprKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|(x, e), f| write!(f, "{}: {}", x.pretty(fmt), e.pretty(fmt)), ", "))
            }
            hir::ExprKind::Tuple(es) => write!(f, "({es})", es = es.all_pretty(", ", fmt)),
            hir::ExprKind::Loop(e) => write!(
                f,
                "loop {{{s1}{e}}}",
                e = e.pretty(fmt),
                s1 = fmt.indent(),
            ),
            hir::ExprKind::Break => write!(f, "break"),
            hir::ExprKind::Unwrap(x0, e0) => write!(f, "unwrap[{}]({})", x0.pretty(fmt), e0.pretty(fmt)),
            hir::ExprKind::Enwrap(x0, e0) => write!(f, "enwrap[{}]({})", x0.pretty(fmt), e0.pretty(fmt)),
            hir::ExprKind::Is(x0, e0) => write!(f, "is[{}]({})", x0.pretty(fmt), e0.pretty(fmt)),
            hir::ExprKind::Var(x) => write!(f, "{}", x.pretty(fmt)),
            hir::ExprKind::Item(x) => write!(f, "{}", x.pretty(fmt)), 
            hir::ExprKind::Err => write!(f, "☇"),
            hir::ExprKind::Return(e) => write!(f, "return {};;", e.pretty(fmt)),
            hir::ExprKind::Todo => write!(f, "???"),
        }?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, hir::LitKind, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(lit, _) = self;
        match lit {
            hir::LitKind::I8(l)   => write!(f, "{}i8", l),
            hir::LitKind::I16(l)  => write!(f, "{}i16", l),
            hir::LitKind::I32(l)  => write!(f, "{}", l),
            hir::LitKind::I64(l)  => write!(f, "{}i64", l),
            hir::LitKind::U8(l)   => write!(f, "{}u8", l),
            hir::LitKind::U16(l)  => write!(f, "{}u16", l),
            hir::LitKind::U32(l)  => write!(f, "{}u32", l),
            hir::LitKind::U64(l)  => write!(f, "{}u64", l),
            hir::LitKind::Bf16(l) => write!(f, "{}bf16", l),
            hir::LitKind::F16(l)  => write!(f, "{}f16", l),
            hir::LitKind::F32(l)  => write!(f, "{}f32", ryu::Buffer::new().format(*l)),
            hir::LitKind::F64(l)  => write!(f, "{}", ryu::Buffer::new().format(*l)),
            hir::LitKind::Bool(l) => write!(f, "{}", l),
            hir::LitKind::Char(l) => write!(f, "'{}'", l),
            hir::LitKind::Str(l)  => write!(f, r#""{}""#, l),
            hir::LitKind::Time(l) => write!(f, "{}", l.as_seconds_f64()),
            hir::LitKind::Unit    => write!(f, "()"),
            hir::LitKind::Err     => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::BinOp, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, fmt) = self;
        match &op.kind {
            hir::BinOpKind::Add  => write!(f, " + "),
            hir::BinOpKind::Sub  => write!(f, " - "),
            hir::BinOpKind::Mul  => write!(f, " * "),
            hir::BinOpKind::Div  => write!(f, " / "),
            hir::BinOpKind::Mod  => write!(f, " % "),
            hir::BinOpKind::Pow  => write!(f, " ** "),
            hir::BinOpKind::Equ  => write!(f, " == "),
            hir::BinOpKind::Neq  => write!(f, " != "),
            hir::BinOpKind::Gt   => write!(f, " > "),
            hir::BinOpKind::Lt   => write!(f, " < "),
            hir::BinOpKind::Geq  => write!(f, " >= "),
            hir::BinOpKind::Leq  => write!(f, " <= "),
            hir::BinOpKind::Or   => write!(f, " or "),
            hir::BinOpKind::And  => write!(f, " and "),
            hir::BinOpKind::Xor  => write!(f, " xor "),
            hir::BinOpKind::Band => write!(f, " band "),
            hir::BinOpKind::Bor  => write!(f, " bor "),
            hir::BinOpKind::Bxor => write!(f, " bxor "),
            hir::BinOpKind::Pipe => write!(f, " |> "),
            hir::BinOpKind::Mut  => write!(f, " := "),
            hir::BinOpKind::Seq  => write!(f, ";{}", fmt),
            hir::BinOpKind::Err  => write!(f, " ☇ "),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::ScalarKind, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, _fmt) = self;
        match kind {
            hir::ScalarKind::Bool => write!(f, "bool"),
            hir::ScalarKind::Char => write!(f, "char"),
            hir::ScalarKind::Bf16 => write!(f, "bf16"),
            hir::ScalarKind::F16  => write!(f, "f16"),
            hir::ScalarKind::F32  => write!(f, "f32"),
            hir::ScalarKind::F64  => write!(f, "f64"),
            hir::ScalarKind::I8   => write!(f, "i8"),
            hir::ScalarKind::I16  => write!(f, "i16"),
            hir::ScalarKind::I32  => write!(f, "i32"),
            hir::ScalarKind::I64  => write!(f, "i64"),
            hir::ScalarKind::U8   => write!(f, "u8"),
            hir::ScalarKind::U16  => write!(f, "u16"),
            hir::ScalarKind::U32  => write!(f, "u32"),
            hir::ScalarKind::U64  => write!(f, "u64"),
            hir::ScalarKind::Null => write!(f, "null"),
            hir::ScalarKind::Str  => write!(f, "str"),
            hir::ScalarKind::Unit => write!(f, "()"),
            hir::ScalarKind::Bot  => todo!(),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Type, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, fmt) = self;
        match &ty.kind {
            hir::TypeKind::Scalar(kind) => write!(f, "{}", kind.pretty(fmt)),
            hir::TypeKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|(x, tv), f| write!(f, "{}: {}", x.pretty(fmt), tv.pretty(fmt)), ", "))
            }
            hir::TypeKind::Nominal(x)     => write!(f, "{}", x.pretty(fmt)),
            hir::TypeKind::Array(ty, sh)  => write!(f, "[{}; {}]", ty.pretty(fmt), sh.pretty(fmt)),
            hir::TypeKind::Stream(ty)     => write!(f, "~{}", ty.pretty(fmt)),
            hir::TypeKind::Map(ty0, ty1)  => write!(f, "{{{} => {}}}", ty0.pretty(fmt), ty1.pretty(fmt)),
            hir::TypeKind::Set(ty)        => write!(f, "{{{}}}", ty.pretty(fmt)),
            hir::TypeKind::Vector(ty)     => write!(f, "[{}]", ty.pretty(fmt)),
            hir::TypeKind::Tuple(tys)     => write!(f, "({})", tys.iter().all_pretty(", ", fmt)),
            hir::TypeKind::Optional(ty)   => write!(f, "{}?", ty.pretty(fmt)),
            hir::TypeKind::Fun(args, ty)  => write!(f, "fun({}) -> {}", args.all_pretty(", ", fmt), ty.pretty(fmt)),
            hir::TypeKind::Err            => write!(f, "☇"),
            hir::TypeKind::Unknown        => write!(f, "?"),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Shape, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(shape, fmt) = self;
        write!(f, "{}", shape.dims.iter().all_pretty(", ", fmt))
    }
}

impl<'i> Display for Pretty<'i, hir::Dim, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(dim, fmt) = self;
        match &dim.kind {
            hir::DimKind::Var(_)       => write!(f, "?"),
            hir::DimKind::Val(v)       => write!(f, "{}", v),
            hir::DimKind::Op(l, op, r) => write!(f, "{}{}{}", l.pretty(fmt), op.pretty(fmt), r.pretty(fmt)),
            hir::DimKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::DimOp, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _fmt) = self;
        match &op.kind {
            hir::DimOpKind::Add => write!(f, "+"),
            hir::DimOpKind::Sub => write!(f, "-"),
            hir::DimOpKind::Mul => write!(f, "*"),
            hir::DimOpKind::Div => write!(f, "/"),
        }
    }
}

// Box implementations (to avoid having to call .as_ref())

impl<'i> Display for Pretty<'i, Box<hir::Expr>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, Box<hir::Dim>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}
