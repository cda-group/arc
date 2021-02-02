#![allow(clippy::useless_format)]
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::shared::New;

use std::fmt::{self, Display, Formatter};

#[derive(New, Copy, Clone)]
pub(crate) struct State<'i> {
    info: &'i Info,
    hir: &'i HIR,
}

pub(crate) fn pretty<'i, 'j, Node>(
    node: &'i Node,
    hir: &'j HIR,
    info: &'j Info,
) -> Pretty<'i, Node, State<'j>> {
    node.to_pretty(State::new(info, hir))
}

impl<'i> Display for Pretty<'i, hir::HIR, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(hir, ctx) = self;
        write!(
            f,
            "{}",
            hir.items
                .iter()
                .filter_map(|x| hir.defs.get(x))
                .all_pretty("\n", ctx)
        );
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, hir::Item, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        match &item.kind {
            hir::ItemKind::Fun(item)     => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::Alias(item)   => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::Enum(item)    => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::Task(item)    => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::State(item)   => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::Extern(item)  => write!(f, "{}", item.pretty(ctx)),
            hir::ItemKind::Variant(item) => write!(f, "{}", item.pretty(ctx)),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Fun, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "fun {id}({params}) -> {ty} {{{s1}{body}{s0}}}",
            id = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            ty = item.rtv.pretty(ctx),
            body = item.body.pretty(&ctx.indent()),
            s0 = ctx,
            s1 = ctx.indent(),
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Extern, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "extern fun {id}({params}) -> {ty};",
            id = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            ty = item.rtv.pretty(ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Param, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(p, ctx) = self;
        match p.kind {
            hir::ParamKind::Var(x) => write!(f, "{}", x.pretty(ctx))?,
            hir::ParamKind::Ignore => write!(f, "_")?,
            hir::ParamKind::Err => write!(f, "☇")?,
        }
        write!(f, ": {}", p.tv.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, hir::Name, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(x, ctx) = self;
        write!(f, "{}", x.id.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, NameId, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(id, ctx) = self;
        write!(f, "{}", ctx.state.info.names.resolve(**id))
    }
}

impl<'i> Display for Pretty<'i, hir::State, State<'_>> {
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

impl<'i> Display for Pretty<'i, hir::Alias, State<'_>> {
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

impl<'i> Display for Pretty<'i, hir::Task, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "task {name}({params}) ({iports}) -> ({oports}) {{{items}{s0}}}",
            name = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            iports = item.iports.iter().all_pretty(", ", ctx),
            oports = item.oports.iter().all_pretty(", ", ctx),
            items = item.items.iter().map_pretty(
                |x, f| write!(
                    f,
                    "{s0}{}",
                    ctx.state.hir.defs.get(x).unwrap().pretty(ctx.indent()),
                    s0 = ctx.indent()
                ),
                ""
            ),
            s0 = ctx,
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Path, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, ctx) = self;
        write!(f, "{}", path.id.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, PathId, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(mut path, ctx) = self;
        let path = ctx.state.info.paths.resolve(*path);
        if let Some(id) = path.pred {
            write!(f, "{}::", id.pretty(ctx))?;
        }
        write!(f, "{}", path.name.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, hir::Enum, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "enum {id} {{{variants}{s0}}}",
            id = item.name.pretty(ctx),
            variants = item.variants.iter().map_pretty(
                |v, f| write!(
                    f,
                    "{s1}{v}",
                    v = ctx.state.hir.defs.get(v).unwrap().pretty(ctx),
                    s1 = ctx.indent()
                ),
                ","
            ),
            s0 = ctx
        )
    }
}

impl<'i> Display for Pretty<'i, hir::Variant, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(variant, ctx) = self;
        let ty = ctx.state.info.types.resolve(variant.tv);
        if let hir::TypeKind::Scalar(hir::ScalarKind::Unit) = ty.kind {
            write!(f, "{}", variant.name.pretty(ctx),)
        } else {
            write!(f, "{}({})", variant.name.pretty(ctx), ty.pretty(ctx))
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Expr, State<'_>> {
    #[allow(clippy::many_single_char_names)]
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(expr, ctx) = self;
        match &expr.kind {
            hir::ExprKind::If(e0, e1, e2) => write!(
                f,
                "if {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                e0 = e0.pretty(&ctx),
                e1 = e1.pretty(&ctx.indent()),
                e2 = e2.pretty(&ctx.indent()),
                s0 = ctx,
                s1 = ctx.indent(),
            ),
            hir::ExprKind::Let(p, e0, e1) => write!(
                f,
                "let {p} = {e0} in{s}{e1}",
                p = p.pretty(ctx),
                e0 = e0.pretty(ctx),
                e1 = e1.pretty(ctx),
                s = ctx
            ),
            hir::ExprKind::Lit(l) => write!(f, "{}", l.pretty(ctx)),
            hir::ExprKind::BinOp(e0, op, e1) => write!(
                f,
                "{e0}{op}{e1}",
                e0 = e0.pretty(ctx),
                op = op.pretty(ctx),
                e1 = e1.pretty(ctx)
            ),
            hir::ExprKind::UnOp(op, e0) => match &op.kind {
                hir::UnOpKind::Not        => write!(f, "not {}", e0.pretty(ctx)),
                hir::UnOpKind::Neg        => write!(f, "-{}", e0.pretty(ctx)),
                hir::UnOpKind::Err        => write!(f, "☇{}", e0.pretty(ctx)),
            },
            hir::ExprKind::Project(e, i) => write!(f, "{}.{}", e.pretty(ctx), i.id),
            hir::ExprKind::Access(e, x) => write!(f, "{}.{}", e.pretty(ctx), x.pretty(ctx)),
            hir::ExprKind::Call(e, es) => write!(f, "{}({})", e.pretty(ctx), es.iter().all_pretty(", ", ctx)),
            hir::ExprKind::Emit(e) => write!(f, "emit {e}", e = e.pretty(ctx)),
            hir::ExprKind::Log(e) => write!(f, "log {e}", e = e.pretty(ctx)),
            hir::ExprKind::Array(es) => write!(f, "[{es}]", es = es.all_pretty(", ", ctx)),
            hir::ExprKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|(x, e), f| write!(f, "{}: {}", x.pretty(ctx), e.pretty(ctx)), ", "))
            }
            hir::ExprKind::Tuple(es) => write!(f, "({es})", es = es.all_pretty(", ", ctx)),
            hir::ExprKind::Loop(e) => write!(
                f,
                "loop {{{s1}{e}}}",
                e = e.pretty(ctx),
                s1 = ctx.indent(),
            ),
            hir::ExprKind::Break => write!(f, "break"),
            hir::ExprKind::Unwrap(x0, e0) => write!(f, "unwrap[{}]({})", x0.pretty(ctx), e0.pretty(ctx)),
            hir::ExprKind::Enwrap(x0, e0) => write!(f, "enwrap[{}]({})", x0.pretty(ctx), e0.pretty(ctx)),
            hir::ExprKind::Is(x0, e0) => write!(f, "is[{}]({})", x0.pretty(ctx), e0.pretty(ctx)),
            hir::ExprKind::Var(x) => write!(f, "{}", x.pretty(ctx)),
            hir::ExprKind::Item(x) => write!(f, "{}", x.pretty(ctx)), 
            hir::ExprKind::Err => write!(f, "☇"),
            hir::ExprKind::Return(e) => write!(f, "return {};;", e.pretty(ctx)),
        }?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, hir::LitKind, State<'_>> {
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

impl<'i> Display for Pretty<'i, hir::BinOp, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, ctx) = self;
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
            hir::BinOpKind::Seq  => write!(f, ";{}", ctx),
            hir::BinOpKind::Err  => write!(f, " ☇ "),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::ScalarKind, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, ctx) = self;
        match kind {
            hir::ScalarKind::Bool => write!(f, "bool"),
            hir::ScalarKind::Char => write!(f, "char"),
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

impl<'i> Display for Pretty<'i, hir::Type, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, ctx) = self;
        match &ty.kind {
            hir::TypeKind::Scalar(kind) => write!(f, "{}", kind.pretty(ctx)),
            hir::TypeKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|(x, tv), f| write!(f, "{}: {}", x.pretty(ctx), tv.pretty(ctx)), ", "))
            }
            hir::TypeKind::Nominal(x)     => write!(f, "{}", x.pretty(ctx)),
            hir::TypeKind::Array(ty, sh)  => write!(f, "[{}; {}]", ty.pretty(ctx), sh.pretty(ctx)),
            hir::TypeKind::Stream(ty)     => write!(f, "~{}", ty.pretty(ctx)),
            hir::TypeKind::Map(ty0, ty1)  => write!(f, "{{{} => {}}}", ty0.pretty(ctx), ty1.pretty(ctx)),
            hir::TypeKind::Set(ty)        => write!(f, "{{{}}}", ty.pretty(ctx)),
            hir::TypeKind::Vector(ty)     => write!(f, "[{}]", ty.pretty(ctx)),
            hir::TypeKind::Tuple(tys)     => write!(f, "({})", tys.iter().all_pretty(", ", ctx)),
            hir::TypeKind::Optional(ty)   => write!(f, "{}?", ty.pretty(ctx)),
            hir::TypeKind::Fun(args, ty)  => write!(f, "fun({}) -> {}", args.all_pretty(", ", ctx), ty.pretty(ctx)),
            hir::TypeKind::Task(ty0, ty1) => write!(f, "({}) => ({})", ty0.all_pretty(", ", ctx), ty1.all_pretty(", ", ctx)),
            hir::TypeKind::Err            => write!(f, "☇"),
            hir::TypeKind::Unknown        => write!(f, "?"),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::Shape, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(shape, ctx) = self;
        write!(f, "{}", shape.dims.iter().all_pretty(", ", ctx))
    }
}

impl<'i> Display for Pretty<'i, hir::Dim, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(dim, ctx) = self;
        match &dim.kind {
            hir::DimKind::Var(_)       => write!(f, "?"),
            hir::DimKind::Val(v)       => write!(f, "{}", v),
            hir::DimKind::Op(l, op, r) => write!(f, "{}{}{}", l.pretty(ctx), op.pretty(ctx), r.pretty(ctx)),
            hir::DimKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, hir::DimOp, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _ctx) = self;
        match &op.kind {
            hir::DimOpKind::Add => write!(f, "+"),
            hir::DimOpKind::Sub => write!(f, "-"),
            hir::DimOpKind::Mul => write!(f, "*"),
            hir::DimOpKind::Div => write!(f, "/"),
        }
    }
}

// Box implementations (to avoid having to call .as_ref())

impl<'i> Display for Pretty<'i, Box<hir::Expr>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, Box<hir::Dim>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}
