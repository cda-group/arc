use crate::compiler::ast;
use crate::compiler::info;
use crate::compiler::info::names::NameId;
use crate::compiler::info::Info;
use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::display::pretty::*;
use crate::compiler::shared::New;

use std::fmt::{self, Display, Formatter};

#[derive(New, Copy, Clone)]
pub(crate) struct State<'i> {
    info: &'i info::Info,
    ast: &'i ast::AST,
}

impl ast::AST {
    pub(crate) fn pretty<'i, 'j, Node>(
        &'j self,
        node: &'i Node,
        info: &'j Info,
    ) -> Pretty<'i, Node, State<'j>> {
        node.to_pretty(State::new(info, self))
    }
}

impl<'i> Display for Pretty<'i, ast::AST, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(prog, ctx) = self;
        write!(f, "{}", prog.modules.values().all_pretty("\n", ctx))
    }
}

impl<'i> Display for Pretty<'i, ast::Module, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(module, ctx) = self;
        write!(f, "{}", module.items.iter().all_pretty("\n", ctx))
    }
}

impl<'i> Display for Pretty<'i, ast::Item, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        match &item.kind {
            ast::ItemKind::Extern(item) => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Fun(item)    => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Alias(item)  => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Use(item)    => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Task(item)   => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Enum(item)   => write!(f, "{}", item.pretty(ctx)),
            ast::ItemKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::TaskItem, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        match &item.kind {
            ast::TaskItemKind::Fun(item)   => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::Alias(item) => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::Use(item)   => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::Enum(item)  => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::On(item)    => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::State(item) => write!(f, "{}", item.pretty(*ctx)),
            ast::TaskItemKind::Err         => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Extern, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "extern fun {id}({params}) -> {ty};",
            id = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            ty = item.return_ty.pretty(ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Fun, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "fun {id}({params}) {{{s1}{body}{s0}}}",
            id = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(", ", ctx),
            body = item.body.pretty(ctx.indent()),
            s0 = ctx,
            s1 = ctx.indent(),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Alias, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "type {id} = {ty}",
            id = item.name.pretty(ctx),
            ty = item.ty.pretty(ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Use, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(f, "use {}", item.path.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, ast::Task, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "task {name}({params}) ({iports}) -> ({oports}) {{{items}{s0}}}",
            name = item.name.pretty(ctx),
            params = item.params.iter().all_pretty(",", ctx),
            iports = item.iports.iter().all_pretty(",", ctx),
            oports = item.oports.iter().all_pretty(",", ctx),
            items = item.items.iter().map_pretty(
                |i, f| write!(f, "{s0}{}", i.pretty(ctx.indent()), s0 = ctx.indent()),
                ""
            ),
            s0 = ctx,
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Enum, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "enum {id} {{ {variants} }}",
            id = item.name.pretty(ctx),
            variants = item.variants.iter().all_pretty(", ", ctx)
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Param, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(param, ctx) = self;
        if let Some(ty) = &param.ty {
            write!(f, "{}: {}", param.pat.pretty(ctx), ty.pretty(ctx))
        } else {
            write!(f, "{}", param.pat.pretty(ctx))
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Variant, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(variant, ctx) = self;
        if let Some(ty) = &variant.ty {
            write!(f, "{}({})", variant.name.pretty(ctx), ty.pretty(ctx))
        } else {
            write!(f, "{}", variant.name.pretty(ctx))
        }
    }
}

impl<'i> Display for Pretty<'i, ast::On, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "on {cases}",
            cases = item.cases.iter().all_pretty(",", ctx)
        )
    }
}

impl<'i> Display for Pretty<'i, ast::State, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, ctx) = self;
        write!(
            f,
            "state {sym} = {expr}",
            sym = item.name.pretty(ctx),
            expr = item.expr.pretty(ctx),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Case, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(case, ctx) = self;
        write!(f, "{} => {}", case.pat.pretty(ctx), case.body.pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, ast::Expr, State<'_>> {
    #[allow(clippy::many_single_char_names)]
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(expr, ctx) = self;
        match &ctx.state.ast.exprs.resolve(expr.id) {
            ast::ExprKind::If(e0, e1, e2) => write!(
                f,
                "if {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                e0 = e0.pretty(&ctx),
                e1 = e1.pretty(&ctx.indent()),
                e2 = e2.pretty(&ctx.indent()),
                s0 = ctx,
                s1 = ctx.indent(),
            ),
            ast::ExprKind::IfLet(p, e0, e1, e2) => write!(
                f,
                "if let {p} = {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                p = p.pretty(ctx),
                e0 = e0.pretty(&ctx),
                e1 = e1.pretty(&ctx.indent()),
                e2 = e2.pretty(&ctx.indent()),
                s0 = ctx,
                s1 = ctx.indent(),
            ),
            ast::ExprKind::For(p, e0, e2) => write!(
                f,
                "for {p} in {e0} {{{s1}{e2}{s0}}}",
                p = p.pretty(&ctx),
                e0 = e0.pretty(&ctx),
                e2 = e2.pretty(&ctx),
                s0 = ctx,
                s1 = ctx.indent()
            ),
            ast::ExprKind::Match(e, cs) => write!(
                f,
                "match {e} {{{cs}{s0}}}{s1}",
                e = e.pretty(&ctx.indent()),
                cs = cs.all_pretty(", ", ctx),
                s0 = ctx,
                s1 = ctx.indent(),
            ),
            ast::ExprKind::Let(p, e0, e1) => write!(
                f,
                "let {p} = {e0} in{s}{e1}",
                p = p.pretty(ctx),
                e0 = e0.pretty(ctx),
                e1 = e1.pretty(ctx),
                s = ctx
            ),
            ast::ExprKind::Lambda(ps, e0) => write!(
                f,
                "|{ps}| {{{s1}{e0}{s0}}}",
                ps = ps.iter().all_pretty(",", ctx),
                e0 = e0.pretty(&ctx.indent()),
                s0 = ctx,
                s1 = ctx.indent(),
            ),
            ast::ExprKind::Lit(l) => write!(f, "{}", l.pretty(ctx)),
            ast::ExprKind::Path(x) => write!(f, "{}", x.pretty(ctx)),
            ast::ExprKind::BinOp(e0, op, e1) => write!(
                f,
                "{e0}{op}{e1}",
                e0 = e0.pretty(ctx),
                op = op.pretty(ctx),
                e1 = e1.pretty(ctx)
            ),
            ast::ExprKind::UnOp(op, e0) => match &op.kind {
                ast::UnOpKind::Not => write!(f, "not {}", e0.pretty(ctx)),
                ast::UnOpKind::Neg => write!(f, "-{}", e0.pretty(ctx)),
                ast::UnOpKind::Err => write!(f, "☇{}", e0.pretty(ctx)),
            },
            ast::ExprKind::Call(e, es) => write!(f, "{}({})", e.pretty(ctx), es.iter().all_pretty(", ", ctx)),
            ast::ExprKind::Cast(e, ty) => write!(f, "{} as {}", e.pretty(ctx), ty.pretty(ctx)),
            ast::ExprKind::Project(e, i) => write!(f, "{}.{}", e.pretty(ctx), i.id),
            ast::ExprKind::Access(e, x) => write!(f, "{}.{}", e.pretty(ctx), x.id.pretty(ctx)),
            ast::ExprKind::Emit(e) => write!(f, "emit {}", e.pretty(ctx)),
            ast::ExprKind::Log(e) => write!(f, "log {}", e.pretty(ctx)),
            ast::ExprKind::Unwrap(x, e) => write!(f, "unwrap[{}]({})", x.pretty(ctx), e.pretty(ctx)),
            ast::ExprKind::Enwrap(x, e) => write!(f, "enwrap[{}]({})", x.pretty(ctx), e.pretty(ctx)),
            ast::ExprKind::Is(x, e) => write!(f, "is[{}]({})",x.pretty(ctx), e.pretty(ctx)),
            ast::ExprKind::Array(es) => write!(f, "[{}]", es.all_pretty(", ", ctx)),
            ast::ExprKind::Struct(fs) => {
                write!(f, "{{ {} }}",
                    fs.map_pretty(|x, f| write!(f, "{}: {}", x.name.pretty(ctx), x.val.pretty(ctx)), ", "))
            }
            ast::ExprKind::Tuple(es) => write!(f, "({es})", es = es.all_pretty(", ", ctx)),
            ast::ExprKind::Loop(e) => write!(
                f,
                "loop {{{s1}{e}}}",
                e = e.pretty(ctx),
                s1 = ctx.indent(),
            ),
            ast::ExprKind::Break => write!(f, "break"),
            ast::ExprKind::Reduce(p, e, r) =>
                write!(f, "reduce {p} = {e} {r}", p=p.pretty(ctx), e=e.pretty(ctx), r=r.pretty(ctx)),
            ast::ExprKind::Err => write!(f, "☇"),
            ast::ExprKind::Return(Some(e)) => write!(f, "return {};;", e.pretty(ctx)),
            ast::ExprKind::Return(None) => write!(f, "return;;"),
        }?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, ast::ReduceKind, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, ctx) = self;
        match kind {
            ast::ReduceKind::Loop(e) => {
                write!(f, "loop {{{s1}{e}}}", e = e.pretty(ctx), s1 = ctx.indent())
            }
            ast::ReduceKind::For(p, e0, e1) => write!(
                f,
                "for {p} in {e0} {{{s1}{e1}}}",
                p = p.pretty(ctx),
                e0 = e0.pretty(ctx),
                e1 = e1.pretty(ctx),
                s1 = ctx.indent()
            ),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::LitKind, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(lit, _) = self;
        match lit {
            ast::LitKind::I8(l)   => write!(f, "{}i8", l),
            ast::LitKind::I16(l)  => write!(f, "{}i16", l),
            ast::LitKind::I32(l)  => write!(f, "{}", l),
            ast::LitKind::I64(l)  => write!(f, "{}i64", l),
            ast::LitKind::U8(l)   => write!(f, "{}u8", l),
            ast::LitKind::U16(l)  => write!(f, "{}u16", l),
            ast::LitKind::U32(l)  => write!(f, "{}u32", l),
            ast::LitKind::U64(l)  => write!(f, "{}u64", l),
            ast::LitKind::F32(l)  => write!(f, "{}f32", ryu::Buffer::new().format(*l)),
            ast::LitKind::F64(l)  => write!(f, "{}", ryu::Buffer::new().format(*l)),
            ast::LitKind::Bool(l) => write!(f, "{}", l),
            ast::LitKind::Char(l) => write!(f, "'{}'", l),
            ast::LitKind::Str(l)  => write!(f, r#""{}""#, l),
            ast::LitKind::Time(l) => write!(f, "{}", l.as_seconds_f64()),
            ast::LitKind::Unit    => write!(f, "()"),
            ast::LitKind::Err     => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::BinOp, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, ctx) = self;
        match &op.kind {
            ast::BinOpKind::Add  => write!(f, " + "),
            ast::BinOpKind::Sub  => write!(f, " - "),
            ast::BinOpKind::Mul  => write!(f, " * "),
            ast::BinOpKind::Div  => write!(f, " / "),
            ast::BinOpKind::Mod  => write!(f, " % "),
            ast::BinOpKind::Pow  => write!(f, " ** "),
            ast::BinOpKind::Equ  => write!(f, " == "),
            ast::BinOpKind::Neq  => write!(f, " != "),
            ast::BinOpKind::Gt   => write!(f, " > "),
            ast::BinOpKind::Lt   => write!(f, " < "),
            ast::BinOpKind::Geq  => write!(f, " >= "),
            ast::BinOpKind::Leq  => write!(f, " <= "),
            ast::BinOpKind::Or   => write!(f, " or "),
            ast::BinOpKind::And  => write!(f, " and "),
            ast::BinOpKind::Xor  => write!(f, " xor "),
            ast::BinOpKind::Band => write!(f, " band "),
            ast::BinOpKind::Bor  => write!(f, " bor "),
            ast::BinOpKind::Bxor => write!(f, " bxor "),
            ast::BinOpKind::Pipe => write!(f, " |> "),
            ast::BinOpKind::Seq  => write!(f, ";{}", ctx),
            ast::BinOpKind::Err  => write!(f, " ☇ "),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Path, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(path, ctx) = self;
        let names = ctx.state.info.paths.resolve(path.id).all_pretty("::", ctx);
        match &path.kind {
            ast::PathKind::Relative => write!(f, "{}", names),
            ast::PathKind::Absolute => write!(f, "::{}", names),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::ScalarKind, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, ctx) = self;
        match kind {
            ast::ScalarKind::Bool => write!(f, "bool"),
            ast::ScalarKind::Char => write!(f, "char"),
            ast::ScalarKind::F32  => write!(f, "f32"),
            ast::ScalarKind::F64  => write!(f, "f64"),
            ast::ScalarKind::I8   => write!(f, "i8"),
            ast::ScalarKind::I16  => write!(f, "i16"),
            ast::ScalarKind::I32  => write!(f, "i32"),
            ast::ScalarKind::I64  => write!(f, "i64"),
            ast::ScalarKind::U8   => write!(f, "u8"),
            ast::ScalarKind::U16  => write!(f, "u16"),
            ast::ScalarKind::U32  => write!(f, "u32"),
            ast::ScalarKind::U64  => write!(f, "u64"),
            ast::ScalarKind::Null => write!(f, "null"),
            ast::ScalarKind::Str  => write!(f, "str"),
            ast::ScalarKind::Unit => write!(f, "()"),
            ast::ScalarKind::Bot  => write!(f, "!"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Type, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, ctx) = self;
        match &ty.kind {
            ast::TypeKind::Scalar(kind) => write!(f, "{}", kind.pretty(ctx)),
            ast::TypeKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|x, f| write!(f, "{}: {}", x.name.pretty(ctx), x.val.pretty(ctx)), ", "))
            }
            ast::TypeKind::Nominal(x)          => write!(f, "{}", x.pretty(ctx)),
            ast::TypeKind::Array(None, sh)     => write!(f, "[{}]", sh.pretty(ctx)),
            ast::TypeKind::Array(Some(ty), sh) => write!(f, "[{}; {}]", ty.pretty(ctx), sh.pretty(ctx)),
            ast::TypeKind::Stream(ty)          => write!(f, "~{}", ty.pretty(ctx)),
            ast::TypeKind::Map(ty0, ty1)       => write!(f, "{{{} => {}}}", ty0.pretty(ctx), ty1.pretty(ctx)),
            ast::TypeKind::Set(ty)             => write!(f, "{{{}}}", ty.pretty(ctx)),
            ast::TypeKind::Vector(ty)          => write!(f, "[{}]", ty.pretty(ctx)),
            ast::TypeKind::Tuple(tys)          => write!(f, "({})", tys.iter().all_pretty(", ", ctx)),
            ast::TypeKind::Optional(ty)        => write!(f, "{}?", ty.pretty(ctx)),
            ast::TypeKind::Fun(args, ty)       => write!(f, "fun({}) -> {}", args.all_pretty(", ", ctx), ty.pretty(ctx)),
            ast::TypeKind::Task(ty0, ty1)      => write!(f, "({}) => ({})", ty0.all_pretty(", ", ctx), ty1.all_pretty(", ", ctx)),
            ast::TypeKind::Err                 => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, NameId, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(id, ctx) = self;
        write!(f, "{}", ctx.state.info.names.resolve(**id))
    }
}

impl<'i> Display for Pretty<'i, ast::Name, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(sym, ctx) = self;
        write!(f, "{}", ctx.state.info.names.resolve(sym.id))
    }
}

impl<'i> Display for Pretty<'i, ast::Shape, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(shape, ctx) = self;
        write!(f, "{}", shape.dims.iter().all_pretty(", ", ctx))
    }
}

impl<'i> Display for Pretty<'i, ast::Dim, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(dim, ctx) = self;
        match &dim.kind {
            ast::DimKind::Var(_)       => write!(f, "?"),
            ast::DimKind::Val(v)       => write!(f, "{}", v),
            ast::DimKind::Op(l, op, r) => write!(f, "{}{}{}", l.pretty(ctx), op.pretty(ctx), r.pretty(ctx)),
            ast::DimKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::DimOp, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _ctx) = self;
        match &op.kind {
            ast::DimOpKind::Add => write!(f, "+"),
            ast::DimOpKind::Sub => write!(f, "-"),
            ast::DimOpKind::Mul => write!(f, "*"),
            ast::DimOpKind::Div => write!(f, "/"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Pat, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(pat, ctx) = self;
        match &pat.kind {
            ast::PatKind::Or(p0, p1)    => write!(f, "{} | {}", p0.pretty(ctx), p1.pretty(ctx)),
            ast::PatKind::Val(l)        => write!(f, "{}", l.pretty(ctx)),
            ast::PatKind::Var(x)        => write!(f, "{}", x.pretty(ctx)),
            ast::PatKind::Tuple(ps)     => write!(f, "({})", ps.all_pretty(", ", ctx)),
            ast::PatKind::Struct(fs)    => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|x, f| if let Some(p) = &x.val {
                                write!(f, "{}: {}", x.name.pretty(ctx), p.pretty(ctx))
                            } else {
                                write!(f, "{}", x.name.pretty(ctx))
                            }, ", "))
            },
            ast::PatKind::Ignore        => write!(f, "_"),
            ast::PatKind::Err           => write!(f, "☇"),
            ast::PatKind::Variant(x, p) => write!(f, "{}({})", x.pretty(ctx), p.pretty(ctx)),
        }
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Expr>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Type>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Pat>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Dim>, State<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, ctx) = self;
        write!(f, "{}", b.as_ref().pretty(ctx))
    }
}
