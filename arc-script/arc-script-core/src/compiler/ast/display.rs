//! AST display utilities.

use crate::compiler::ast;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;
use crate::compiler::pretty::*;
use arc_script_core_shared::New;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

/// Struct which contains all the information needed for pretty printing `AST:s`.
#[derive(New, Copy, Clone)]
pub(crate) struct Context<'i> {
    info: &'i Info,
    ast: &'i ast::AST,
}

impl ast::AST {
    /// Returns a struct which can be used to pretty print `Node`.
    pub(crate) fn pretty<'i, 'j, Node>(
        &'j self,
        node: &'i Node,
        info: &'j Info,
    ) -> Pretty<'i, Node, Context<'j>> {
        node.to_pretty(Context::new(info, self))
    }
}

impl<'i> Display for Pretty<'i, ast::AST, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(prog, fmt) = self;
        write!(f, "{}", prog.modules.values().all_pretty("\n", fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::Module, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(module, fmt) = self;
        write!(f, "{}", module.items.iter().all_pretty("\n", fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::Item, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match &item.kind {
            ast::ItemKind::Extern(item) => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Fun(item)    => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Alias(item)  => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Use(item)    => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Task(item)   => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Enum(item)   => write!(f, "{}", item.pretty(fmt)),
            ast::ItemKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::TaskItem, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match &item.kind {
            ast::TaskItemKind::Fun(item)    => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::Extern(item) => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::Alias(item)  => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::Use(item)    => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::Enum(item)   => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::On(item)     => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::State(item)  => write!(f, "{}", item.pretty(fmt)),
            ast::TaskItemKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Extern, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "extern fun {id}({params}) -> {ty};",
            id = item.name.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            ty = item.return_ty.pretty(fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Fun, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "fun {id}({params}) {{{s1}{body}{s0}}}",
            id = item.name.pretty(fmt),
            params = item.params.iter().all_pretty(", ", fmt),
            body = item.body.pretty(fmt.indent()),
            s0 = fmt,
            s1 = fmt.indent(),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Alias, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "type {id} = {ty}",
            id = item.name.pretty(fmt),
            ty = item.ty.pretty(fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Use, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(f, "use {}", item.path.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::Task, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "task {name}({params}) ({ihub}) -> ({ohub}) {{{items}{s0}}}",
            name = item.name.pretty(fmt),
            params = item.params.iter().all_pretty(",", fmt),
            ihub = item.ihub.pretty(fmt),
            ohub = item.ohub.pretty(fmt),
            items = item.items.iter().map_pretty(
                |i, f| write!(f, "{s0}{}", i.pretty(fmt.indent()), s0 = fmt.indent()),
                ""
            ),
            s0 = fmt,
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Hub, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        match &item.kind {
            ast::HubKind::Tagged(ports) => {
                write!(f, "({})", ports.iter().all_pretty(", ", fmt))
            }
            ast::HubKind::Single(t) => {
                write!(f, "({})", t.pretty(fmt))
            }
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Enum, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "enum {id} {{ {variants} }}",
            id = item.name.pretty(fmt),
            variants = item.variants.iter().all_pretty(", ", fmt)
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Param, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(param, fmt) = self;
        if let Some(ty) = &param.ty {
            write!(f, "{}: {}", param.pat.pretty(fmt), ty.pretty(fmt))
        } else {
            write!(f, "{}", param.pat.pretty(fmt))
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Variant, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(variant, fmt) = self;
        if let Some(ty) = &variant.ty {
            write!(f, "{}({})", variant.name.pretty(fmt), ty.pretty(fmt))
        } else {
            write!(f, "{}", variant.name.pretty(fmt))
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Port, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(port, fmt) = self;
        write!(f, "{}({})", port.name.pretty(fmt), port.ty.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::On, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "on {cases}",
            cases = item.cases.iter().all_pretty(",", fmt)
        )
    }
}

impl<'i> Display for Pretty<'i, ast::State, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(item, fmt) = self;
        write!(
            f,
            "state {sym} = {expr}",
            sym = item.name.pretty(fmt),
            expr = item.expr.pretty(fmt),
        )
    }
}

impl<'i> Display for Pretty<'i, ast::Case, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(case, fmt) = self;
        write!(f, "{} => {}", case.pat.pretty(fmt), case.body.pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::Expr, Context<'_>> {
    #[allow(clippy::many_single_char_names)]
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(expr, fmt) = self;
        match &fmt.ctx.ast.exprs.resolve(expr.id) {
            ast::ExprKind::If(e0, e1, e2) => write!(
                f,
                "if {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                e0 = e0.pretty(&fmt),
                e1 = e1.pretty(&fmt.indent()),
                e2 = e2.pretty(&fmt.indent()),
                s0 = fmt,
                s1 = fmt.indent(),
            ),
            ast::ExprKind::IfLet(p, e0, e1, e2) => write!(
                f,
                "if let {p} = {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                p = p.pretty(fmt),
                e0 = e0.pretty(&fmt),
                e1 = e1.pretty(&fmt.indent()),
                e2 = e2.pretty(&fmt.indent()),
                s0 = fmt,
                s1 = fmt.indent(),
            ),
            ast::ExprKind::For(p, e0, e2) => write!(
                f,
                "for {p} in {e0} {{{s1}{e2}{s0}}}",
                p = p.pretty(&fmt),
                e0 = e0.pretty(&fmt),
                e2 = e2.pretty(&fmt),
                s0 = fmt,
                s1 = fmt.indent()
            ),
            ast::ExprKind::Match(e, cs) => write!(
                f,
                "match {e} {{{cs}{s0}}}{s1}",
                e = e.pretty(&fmt.indent()),
                cs = cs.all_pretty(", ", fmt),
                s0 = fmt,
                s1 = fmt.indent(),
            ),
            ast::ExprKind::Let(p, e0, e1) => write!(
                f,
                "let {p} = {e0} in{s}{e1}",
                p = p.pretty(fmt),
                e0 = e0.pretty(fmt),
                e1 = e1.pretty(fmt),
                s = fmt
            ),
            ast::ExprKind::Lambda(ps, e0) => write!(
                f,
                "|{ps}| {{{s1}{e0}{s0}}}",
                ps = ps.iter().all_pretty(",", fmt),
                e0 = e0.pretty(&fmt.indent()),
                s0 = fmt,
                s1 = fmt.indent(),
            ),
            ast::ExprKind::Lit(l) => write!(f, "{}", l.pretty(fmt)),
            ast::ExprKind::Path(x) => write!(f, "{}", x.pretty(fmt)),
            ast::ExprKind::BinOp(e0, op, e1) => write!(
                f,
                "{e0}{op}{e1}",
                e0 = e0.pretty(fmt),
                op = op.pretty(fmt),
                e1 = e1.pretty(fmt)
            ),
            ast::ExprKind::UnOp(op, e0) => match &op.kind {
                ast::UnOpKind::Boxed => write!(f, "box {}", e0.pretty(fmt)),
                ast::UnOpKind::Not => write!(f, "not {}", e0.pretty(fmt)),
                ast::UnOpKind::Neg => write!(f, "-{}", e0.pretty(fmt)),
                ast::UnOpKind::Err => write!(f, "☇{}", e0.pretty(fmt)),
            },
            ast::ExprKind::Call(e, es) => write!(f, "{}({})", e.pretty(fmt), es.iter().all_pretty(", ", fmt)),
            ast::ExprKind::Cast(e, ty) => write!(f, "{} as {}", e.pretty(fmt), ty.pretty(fmt)),
            ast::ExprKind::Project(e, i) => write!(f, "{}.{}", e.pretty(fmt), i.id),
            ast::ExprKind::Access(e, x) => write!(f, "{}.{}", e.pretty(fmt), x.id.pretty(fmt)),
            ast::ExprKind::Emit(e) => write!(f, "emit {}", e.pretty(fmt)),
            ast::ExprKind::Log(e) => write!(f, "log {}", e.pretty(fmt)),
            ast::ExprKind::Unwrap(x, e) => write!(f, "unwrap[{}]({})", x.pretty(fmt), e.pretty(fmt)),
            ast::ExprKind::Enwrap(x, e) => write!(f, "enwrap[{}]({})", x.pretty(fmt), e.pretty(fmt)),
            ast::ExprKind::Is(x, e) => write!(f, "is[{}]({})",x.pretty(fmt), e.pretty(fmt)),
            ast::ExprKind::Array(es) => write!(f, "[{}]", es.all_pretty(", ", fmt)),
            ast::ExprKind::Struct(fs) => {
                write!(f, "{{ {} }}",
                    fs.map_pretty(|x, f| write!(f, "{}: {}", x.name.pretty(fmt), x.val.pretty(fmt)), ", "))
            }
            ast::ExprKind::Tuple(es) => write!(f, "({es})", es = es.all_pretty(", ", fmt)),
            ast::ExprKind::Loop(e) => write!(
                f,
                "loop {{{s1}{e}}}",
                e = e.pretty(fmt),
                s1 = fmt.indent(),
            ),
            ast::ExprKind::Break => write!(f, "break"),
            ast::ExprKind::Reduce(p, e, r) =>
                write!(f, "reduce {p} = {e} {r}", p=p.pretty(fmt), e=e.pretty(fmt), r=r.pretty(fmt)),
            ast::ExprKind::Err => write!(f, "☇"),
            ast::ExprKind::Return(Some(e)) => write!(f, "return {};;", e.pretty(fmt)),
            ast::ExprKind::Return(None) => write!(f, "return;;"),
            ast::ExprKind::Todo => write!(f, "???"),
        }?;
        Ok(())
    }
}

impl<'i> Display for Pretty<'i, ast::ReduceKind, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, fmt) = self;
        match kind {
            ast::ReduceKind::Loop(e) => {
                write!(f, "loop {{{s1}{e}}}", e = e.pretty(fmt), s1 = fmt.indent())
            }
            ast::ReduceKind::For(p, e0, e1) => write!(
                f,
                "for {p} in {e0} {{{s1}{e1}}}",
                p = p.pretty(fmt),
                e0 = e0.pretty(fmt),
                e1 = e1.pretty(fmt),
                s1 = fmt.indent()
            ),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::LitKind, Context<'_>> {
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
            ast::LitKind::Bf16(l) => write!(f, "{}bf16", l),
            ast::LitKind::F16(l)  => write!(f, "{}f16", l),
            ast::LitKind::F32(l)  => write!(f, "{}f32", ryu::Buffer::new().format(*l)),
            ast::LitKind::F64(l)  => write!(f, "{}", ryu::Buffer::new().format(*l)),
            ast::LitKind::Bool(l) => write!(f, "{}", l),
            ast::LitKind::Char(l) => write!(f, "'{}'", l),
            ast::LitKind::Str(l)  => write!(f, r#""{}""#, l),
            ast::LitKind::Time(l) => write!(f, "{}", l.as_seconds_f64()),
            ast::LitKind::Unit    => write!(f, "unit"),
            ast::LitKind::Err     => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::BinOp, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, fmt) = self;
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
            ast::BinOpKind::By   => write!(f, " by "),
            ast::BinOpKind::Pipe => write!(f, " |> "),
            ast::BinOpKind::Mut  => write!(f, " = "),
            ast::BinOpKind::Seq  => write!(f, ";{}", fmt),
            ast::BinOpKind::Err  => write!(f, " ☇ "),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Path, Context<'_>> {
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

impl<'i> Display for Pretty<'i, ast::ScalarKind, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(kind, _fmt) = self;
        match kind {
            ast::ScalarKind::Bool => write!(f, "bool"),
            ast::ScalarKind::Char => write!(f, "char"),
            ast::ScalarKind::Bf16 => write!(f, "bf16"),
            ast::ScalarKind::F16  => write!(f, "f16"),
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
            ast::ScalarKind::Unit => write!(f, "unit"),
            ast::ScalarKind::Bot  => write!(f, "!"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Type, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, fmt) = self;
        match &ty.kind {
            ast::TypeKind::Scalar(kind) => write!(f, "{}", kind.pretty(fmt)),
            ast::TypeKind::Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|x, f| write!(f, "{}: {}", x.name.pretty(fmt), x.val.pretty(fmt)), ", "))
            }
            ast::TypeKind::Nominal(x)          => write!(f, "{}", x.pretty(fmt)),
            ast::TypeKind::Array(None, sh)     => write!(f, "[{}]", sh.pretty(fmt)),
            ast::TypeKind::Array(Some(ty), sh) => write!(f, "[{}; {}]", ty.as_ref().pretty(fmt), sh.pretty(fmt)),
            ast::TypeKind::Stream(ty)          => write!(f, "~{}", ty.pretty(fmt)),
            ast::TypeKind::Map(ty0, ty1)       => write!(f, "{{{} => {}}}", ty0.pretty(fmt), ty1.pretty(fmt)),
            ast::TypeKind::Set(ty)             => write!(f, "{{{}}}", ty.pretty(fmt)),
            ast::TypeKind::Vector(ty)          => write!(f, "[{}]", ty.pretty(fmt)),
            ast::TypeKind::Tuple(tys)          => write!(f, "({})", tys.iter().all_pretty(", ", fmt)),
            ast::TypeKind::Optional(ty)        => write!(f, "{}?", ty.pretty(fmt)),
            ast::TypeKind::Fun(args, ty)       => write!(f, "fun({}) -> {}", args.all_pretty(", ", fmt), ty.pretty(fmt)),
            ast::TypeKind::Boxed(ty)           => write!(f, "box {}", ty.pretty(fmt)),
            ast::TypeKind::By(ty0, ty1)        => write!(f, "{} by {}", ty0.pretty(fmt), ty1.pretty(fmt)),
            ast::TypeKind::Err                 => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, NameId, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(id, fmt) = self;
        write!(f, "{}", fmt.ctx.info.names.resolve(**id))
    }
}

impl<'i> Display for Pretty<'i, ast::Name, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(sym, fmt) = self;
        write!(f, "{}", fmt.ctx.info.names.resolve(sym.id))
    }
}

impl<'i> Display for Pretty<'i, ast::Shape, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(shape, fmt) = self;
        write!(f, "{}", shape.dims.iter().all_pretty(", ", fmt))
    }
}

impl<'i> Display for Pretty<'i, ast::Dim, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(dim, fmt) = self;
        match &dim.kind {
            ast::DimKind::Var(_)       => write!(f, "?"),
            ast::DimKind::Val(v)       => write!(f, "{}", v),
            ast::DimKind::Op(l, op, r) => write!(f, "{}{}{}", l.pretty(fmt), op.pretty(fmt), r.pretty(fmt)),
            ast::DimKind::Err          => write!(f, "☇"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::DimOp, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _fmt) = self;
        match &op.kind {
            ast::DimOpKind::Add => write!(f, "+"),
            ast::DimOpKind::Sub => write!(f, "-"),
            ast::DimOpKind::Mul => write!(f, "*"),
            ast::DimOpKind::Div => write!(f, "/"),
        }
    }
}

impl<'i> Display for Pretty<'i, ast::Pat, Context<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(pat, fmt) = self;
        match &pat.kind {
            ast::PatKind::Or(p0, p1)    => write!(f, "{} | {}", p0.pretty(fmt), p1.pretty(fmt)),
            ast::PatKind::Val(l)        => write!(f, "{}", l.pretty(fmt)),
            ast::PatKind::Var(x)        => write!(f, "{}", x.pretty(fmt)),
            ast::PatKind::Tuple(ps)     => write!(f, "({})", ps.all_pretty(", ", fmt)),
            ast::PatKind::Struct(fs)    => {
                write!(f, "{{ {fs} }}",
                    fs = fs.map_pretty(|x, f| if let Some(p) = &x.val {
                                write!(f, "{}: {}", x.name.pretty(fmt), p.pretty(fmt))
                            } else {
                                write!(f, "{}", x.name.pretty(fmt))
                            }, ", "))
            },
            ast::PatKind::Ignore        => write!(f, "_"),
            ast::PatKind::Err           => write!(f, "☇"),
            ast::PatKind::Variant(x, p) => write!(f, "{}({})", x.pretty(fmt), p.pretty(fmt)),
        }
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Expr>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Type>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Pat>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}

impl<'i> Display for Pretty<'i, Box<ast::Dim>, Context<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(b, fmt) = self;
        write!(f, "{}", b.as_ref().pretty(fmt))
    }
}
