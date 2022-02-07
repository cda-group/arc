//! AST display utilities.

#[path = "../pretty.rs"]
#[macro_use]
pub(crate) mod pretty;

use pretty::*;

use crate::ast;
use crate::info::modes::Verbosity;
use crate::info::names::NameId;
use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

/// Struct which contains all the information needed for pretty printing `AST:s`.
#[derive(New, Copy, Clone, Shrinkwrap)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i Info,
    ast: &'i ast::AST,
}

impl ast::AST {
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

    ast::AST => write!(w, "{}", node.modules.values().all_pretty("\n", fmt)),
    ast::Module => write!(w, "{}", node.items.iter().all_pretty("\n", fmt)),
    Vec<ast::Item> => write!(w, "{}", node.iter().map_pretty(|item, w| writeln!(w, "{}", item.pretty(fmt)), "")),
    ast::Item => write!(w, "{}", node.kind.pretty(fmt)),
    ast::ItemKind => match node {
        ast::ItemKind::ExternFun(item)  => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::ExternType(item) => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::Fun(item)        => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::TypeAlias(item)  => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::Use(item)        => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::Task(item)       => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::Enum(item)       => write!(w, "{}", item.pretty(fmt)),
        ast::ItemKind::Assign(item)     => write!(w, "{};", item.pretty(fmt)),
        ast::ItemKind::Err              => write!(w, "☇"),
    },
    Vec<ast::TaskItem> => write!(w, "{}", node.iter().map_pretty(|item, w| writeln!(w, "{}", item.pretty(fmt)), "")),
    ast::TaskItem => write!(w, "{}", node.kind.pretty(fmt)),
    ast::TaskItemKind => match node {
        ast::TaskItemKind::Fun(item)       => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::ExternFun(item) => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::TypeAlias(item) => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::Use(item)       => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::Enum(item)      => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::Stmt(item)      => write!(w, "{}", item.pretty(fmt)),
        ast::TaskItemKind::Err             => write!(w, "☇"),
    },
    Vec<ast::ExternTypeItem> => write!(w, "{}", node.iter().map_pretty(|item, w| writeln!(w, "{}", item.pretty(fmt)), "")),
    ast::ExternTypeItem => write!(w, "{}", node.kind.pretty(fmt)),
    ast::ExternTypeItemKind => match node {
        ast::ExternTypeItemKind::FunDecl(item) => write!(w, "{}", item.pretty(fmt)),
        ast::ExternTypeItemKind::Err           => write!(w, "☇"),
    },
    ast::ExternType => write!(w, "extern type {id}({params}) {{{funs}{s0}}}",
        id = node.name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        funs = node.items.iter().map_pretty(|f, w| write!(w, "{}{}", fmt.indent(), f.pretty(fmt)), ", "),
        s0 = fmt,
    ),
    ast::FunDecl => write!(w, "fun {}({}): {}", node.name.pretty(fmt), node.params.pretty(fmt), node.rt.pretty(fmt)),
    ast::TypeAlias => write!(w, "type {id} = {ty}", id = node.name.pretty(fmt), ty = node.t.pretty(fmt)),
    ast::Use => write!(w, "use {}", node.path.pretty(fmt)),
    ast::Assign => write!(w, "{kind} {param} = {expr}",
        kind = node.kind.pretty(fmt),
        param = node.param.pretty(fmt),
        expr = node.expr.pretty(fmt)
    ),
    ast::MutKind => match node {
        ast::MutKind::Immutable => write!(w, "val"),
        ast::MutKind::Mutable => write!(w, "var"),
    },
    ast::Fun => write!(w, "fun {id}({params}){rty} {body}",
        id = node.name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        body = node.block.pretty(fmt),
        rty = node.rt.iter().map_pretty(|t, w| write!(w, ": {}", t.pretty(fmt)), ""),
    ),
    ast::ExternFun => write!(w, "extern {};", node.decl.pretty(fmt)),
    ast::Task => write!(w, "task {name}({params}) {iinterface} -> {ointerface} {{{items}{s0}}}",
        name = node.name.pretty(fmt),
        params = node.params.iter().all_pretty(", ", fmt),
        iinterface = node.iinterface.pretty(fmt),
        ointerface = node.ointerface.pretty(fmt),
        items = node.items.iter().map_pretty(
            |i, w| write!(w, "{s0}{}", i.pretty(fmt.indent()), s0 = fmt.indent()),
            ""
        ),
        s0 = fmt,
    ),
    ast::Interface => match &node.kind {
        ast::InterfaceKind::Tagged(ps) => write!(w, "{}", ps.iter().all_pretty(", ", fmt)),
        ast::InterfaceKind::Single(t) => write!(w, "{}", t.pretty(fmt)),
    },
    ast::Enum => write!(w, "enum {id} {{ {variants} }}",
        id = node.name.pretty(fmt),
        variants = node.variants.iter().all_pretty(", ", fmt)
    ),
    Vec<ast::Param> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    ast::Param => {
        if let Some(ty) = &node.t {
            write!(w, "{}: {}", node.pat.pretty(fmt), ty.pretty(fmt))
        } else {
            write!(w, "{}", node.pat.pretty(fmt))
        }
    },
    ast::Variant => {
        if let Some(ty) = &node.t {
            write!(w, "{}({})", node.name.pretty(fmt), ty.pretty(fmt))
        } else {
            write!(w, "{}", node.name.pretty(fmt))
        }
    },
    ast::Port => write!(w, "{}({})", node.name.pretty(fmt), node.t.pretty(fmt)),
    Vec<ast::Case> => {
        if let [case] = node.as_slice() {
            write!(w, "on {case}", case = case.pretty(fmt))
        } else {
            write!(w, "on {cases}{s0}",
                cases = node.iter().map_pretty(|c, w| write!(w, "{}{}", fmt.indent(), c.pretty(fmt)), ","),
                s0 = fmt
            )
        }
    },
    ast::Case => write!(w, "{} => {}", node.pat.pretty(fmt), node.body.pretty(fmt)),
    ast::Stmt => match &node.kind {
        ast::StmtKind::Empty => write!(w, ";"),
        ast::StmtKind::Assign(v) => write!(w, "{};{}", v.pretty(fmt), fmt),
        ast::StmtKind::Expr(e) => write!(w, "{};{}", e.pretty(fmt), fmt),
    },
    Vec<ast::Stmt> => write!(w, "{}", node.all_pretty("", fmt)),
    ast::Block => write!(w, "{{{s1}{stmts}{expr}{s0}}}",
        stmts = node.stmts.pretty(fmt.indent()),
        expr = node.expr.pretty(fmt.indent()),
        s0 = fmt,
        s1 = fmt.indent()
    ),
    ast::Expr => {
        let kind = fmt.ast.exprs.resolve(node);
        if fmt.mode.verbosity >= Verbosity::Debug {
            write!(w, "({})", kind.pretty(fmt))?;
        } else {
            write!(w, "{}", kind.pretty(fmt))?;
        }
    },
    Option<ast::Expr> => {
        if let Some(e) = node {
            write!(w, "{}", e.pretty(fmt))?;
        }
    },
    Vec<ast::Expr> => write!(w, "{}", node.iter().all_pretty(", ", fmt)),
    ast::ExprKind => match node {
        ast::ExprKind::On(cases) => write!(w, "{}", cases.pretty(fmt)),
        ast::ExprKind::Return(Some(e)) => write!(w, "return {}", e.pretty(fmt)),
        ast::ExprKind::Return(None) => write!(w, "return"),
        ast::ExprKind::Break(Some(e)) => write!(w, "break {}", e.pretty(fmt)),
        ast::ExprKind::Break(None) => write!(w, "break"),
        ast::ExprKind::Continue => write!(w, "continue"),
        ast::ExprKind::After(e0, e1) => write!(
            w,
            "after {e0} {e1}",
            e0 = e0.pretty(fmt),
            e1 = e1.pretty(fmt),
        ),
        ast::ExprKind::Every(e0, e1) => write!(
            w,
            "every {e0} {e1}",
            e0 = e0.pretty(fmt),
            e1 = e1.pretty(fmt),
        ),
        ast::ExprKind::Block(b) => write!(w, "{}", b.pretty(fmt)),
        ast::ExprKind::If(e0, e1, Some(e2)) => write!(
            w,
            "if {e0} {e1} else {e2}",
            e0 = e0.pretty(fmt),
            e1 = e1.pretty(fmt),
            e2 = e2.pretty(fmt),
        ),
        ast::ExprKind::IfAssign(l, e0, Some(e1)) => write!(
            w,
            "if {l} {e0} else {e1}",
            l = l.pretty(fmt),
            e0 = e0.pretty(fmt),
            e1 = e1.pretty(fmt),
        ),
        ast::ExprKind::If(e0, e1, None) => write!(
            w,
            "if {e0} {e1}",
            e0 = e0.pretty(fmt),
            e1 = e1.pretty(fmt),
        ),
        ast::ExprKind::IfAssign(l, e, None) => write!(
            w,
            "if {l} {e}",
            l = l.pretty(fmt),
            e = e.pretty(fmt),
        ),
        ast::ExprKind::For(p, e0, e2) => write!(
            w,
            "for {p} in {e0} {e2}",
            p = p.pretty(fmt),
            e0 = e0.pretty(fmt),
            e2 = e2.pretty(fmt),
        ),
        ast::ExprKind::Match(e, cs) => write!(
            w,
            "match {e} {{{cs}{s0}}}{s1}",
            e = e.pretty(fmt.indent()),
            cs = cs.all_pretty(", ", fmt),
            s0 = fmt,
            s1 = fmt.indent(),
        ),
        ast::ExprKind::Lambda(ps, e0) => write!(
            w,
            "fun({ps}): {e0}",
            ps = ps.iter().all_pretty(",", fmt),
            e0 = e0.pretty(fmt.indent()),
        ),
        ast::ExprKind::Lit(l) => write!(w, "{}", l.pretty(fmt)),
        ast::ExprKind::Path(x, Some(ts)) => write!(w, "{}::[{}]", x.pretty(fmt), ts.iter().all_pretty(", ", fmt)),
        ast::ExprKind::Path(x, None) => write!(w, "{}", x.pretty(fmt)),
        ast::ExprKind::BinOp(e0, op, e1) => write!(
            w,
            "{e0}{op}{e1}",
            e0 = e0.pretty(fmt),
            op = op.pretty(fmt),
            e1 = e1.pretty(fmt)
        ),
        ast::ExprKind::UnOp(op, e) => match &op.kind {
            ast::UnOpKind::Not => write!(w, "not {}", e.pretty(fmt)),
            ast::UnOpKind::Neg => write!(w, "-{}", e.pretty(fmt)),
            ast::UnOpKind::Err => write!(w, "☇{}", e.pretty(fmt)),
        },
        ast::ExprKind::Call(e, es) => write!(w, "{}({})", e.pretty(fmt), es.pretty(fmt)),
        ast::ExprKind::Invoke(e, x, es) => write!(w, "{}.{}({})", e.pretty(fmt), x.pretty(fmt), es.pretty(fmt)),
        ast::ExprKind::Select(e, es) => write!(w, "{}[{}]", e.pretty(fmt), es.pretty(fmt)),
        ast::ExprKind::Cast(e, ty) => write!(w, "{} as {}", e.pretty(fmt), ty.pretty(fmt)),
        ast::ExprKind::Project(e, i) => write!(w, "{}.{}", e.pretty(fmt), i.id),
        ast::ExprKind::Access(e, x) => write!(w, "{}.{}", e.pretty(fmt), x.id.pretty(fmt)),
        ast::ExprKind::Emit(e) => write!(w, "emit {}", e.pretty(fmt)),
        ast::ExprKind::Log(e) => write!(w, "log {}", e.pretty(fmt)),
        ast::ExprKind::Unwrap(x, e) => write!(w, "unwrap[{}]({})", x.pretty(fmt), e.pretty(fmt)),
        ast::ExprKind::Enwrap(x, e) => write!(w, "enwrap[{}]({})", x.pretty(fmt), e.pretty(fmt)),
        ast::ExprKind::Is(x, e) => write!(w, "is[{}]({})",x.pretty(fmt), e.pretty(fmt)),
        ast::ExprKind::Array(es) => write!(w, "[{}]", es.all_pretty(", ", fmt)),
        ast::ExprKind::Struct(fs) => {
            write!(w, "{{ {} }}",
                fs.map_pretty(|x, w| write!(w, "{}: {}", x.name.pretty(fmt), x.val.pretty(fmt)), ", "))
        }
        ast::ExprKind::Tuple(es) => write!(w, "({es})", es = es.all_pretty(", ", fmt)),
        ast::ExprKind::Loop(e) => write!(
            w,
            "loop {{{s1}{e}}}",
            e = e.pretty(fmt),
            s1 = fmt.indent(),
        ),
        ast::ExprKind::Err => write!(w, "☇"),
    },
    ast::LitKind =>
        match node {
            ast::LitKind::I8(l)       => write!(w, "{}i8", l),
            ast::LitKind::I16(l)      => write!(w, "{}i16", l),
            ast::LitKind::I32(l)      => write!(w, "{}", l),
            ast::LitKind::I64(l)      => write!(w, "{}i64", l),
            ast::LitKind::U8(l)       => write!(w, "{}u8", l),
            ast::LitKind::U16(l)      => write!(w, "{}u16", l),
            ast::LitKind::U32(l)      => write!(w, "{}u32", l),
            ast::LitKind::U64(l)      => write!(w, "{}u64", l),
            ast::LitKind::F32(l)      => write!(w, "{}f32", ryu::Buffer::new().format(*l)),
            ast::LitKind::F64(l)      => write!(w, "{}", ryu::Buffer::new().format(*l)),
            ast::LitKind::Bool(l)     => write!(w, "{}", l),
            ast::LitKind::Char(l)     => write!(w, "'{}'", l),
            ast::LitKind::Str(l)      => write!(w, r#""{}""#, l),
            ast::LitKind::Duration(l) => write!(w, "{}", l.as_seconds_f64()),
            ast::LitKind::DateTime(l) => write!(w, "{}", l),
            ast::LitKind::Unit        => write!(w, "unit"),
            ast::LitKind::Err         => write!(w, "☇"),
        },
    ast::BinOp => write!(w, "{}", node.kind.pretty(fmt)),
    ast::BinOpKind => match node {
        ast::BinOpKind::Add   => write!(w, " + "),
        ast::BinOpKind::Sub   => write!(w, " - "),
        ast::BinOpKind::Mul   => write!(w, " * "),
        ast::BinOpKind::Div   => write!(w, " / "),
        ast::BinOpKind::Mod   => write!(w, " % "),
        ast::BinOpKind::Pow   => write!(w, " ** "),
        ast::BinOpKind::Equ   => write!(w, " == "),
        ast::BinOpKind::Neq   => write!(w, " != "),
        ast::BinOpKind::Gt    => write!(w, " > "),
        ast::BinOpKind::Lt    => write!(w, " < "),
        ast::BinOpKind::Geq   => write!(w, " >= "),
        ast::BinOpKind::Leq   => write!(w, " <= "),
        ast::BinOpKind::Or    => write!(w, " or "),
        ast::BinOpKind::And   => write!(w, " and "),
        ast::BinOpKind::Xor   => write!(w, " xor "),
        ast::BinOpKind::Band  => write!(w, " band "),
        ast::BinOpKind::Bor   => write!(w, " bor "),
        ast::BinOpKind::Bxor  => write!(w, " bxor "),
        ast::BinOpKind::By    => write!(w, " by "),
        ast::BinOpKind::In    => write!(w, " in "),
        ast::BinOpKind::NotIn => write!(w, " not in "),
        ast::BinOpKind::Pipe  => write!(w, " | "),
        ast::BinOpKind::Mut   => write!(w, " = "),
        ast::BinOpKind::RExc  => write!(w, ".."),
        ast::BinOpKind::RInc  => write!(w, "..="),
        ast::BinOpKind::Err   => write!(w, " ☇ "),
    },
    ast::Path => write!(w, "{}", node.id.pretty(fmt)),
    ast::PathId => {
        let path = fmt.paths.resolve(node);
        if let Some(id) = path.pred {
            write!(w, "{}::", id.pretty(fmt))?;
        }
        write!(w, "{}", path.name.pretty(fmt))
    },
    ast::ScalarKind => match node {
        ast::ScalarKind::Bool      => write!(w, "bool"),
        ast::ScalarKind::Char      => write!(w, "char"),
        ast::ScalarKind::F32       => write!(w, "f32"),
        ast::ScalarKind::F64       => write!(w, "f64"),
        ast::ScalarKind::I8        => write!(w, "i8"),
        ast::ScalarKind::I16       => write!(w, "i16"),
        ast::ScalarKind::I32       => write!(w, "i32"),
        ast::ScalarKind::I64       => write!(w, "i64"),
        ast::ScalarKind::U8        => write!(w, "u8"),
        ast::ScalarKind::U16       => write!(w, "u16"),
        ast::ScalarKind::U32       => write!(w, "u32"),
        ast::ScalarKind::U64       => write!(w, "u64"),
        ast::ScalarKind::Str       => write!(w, "str"),
        ast::ScalarKind::Unit      => write!(w, "unit"),
        ast::ScalarKind::Size      => write!(w, "usize"),
        ast::ScalarKind::DateTime  => write!(w, "time"),
        ast::ScalarKind::Duration  => write!(w, "duration"),
    },
    ast::Type => {
        let kind = fmt.ast.types.resolve(node);
        if fmt.mode.verbosity >= Verbosity::Debug {
            write!(w, "({})", kind.pretty(fmt))?;
        } else {
            write!(w, "{}", kind.pretty(fmt))?;
        }
    },
    ast::TypeKind => match node {
        ast::TypeKind::Scalar(kind) => write!(w, "{}", kind.pretty(fmt)),
        ast::TypeKind::Struct(fs) => {
            write!(w, "{{ {fs} }}",
                fs = fs.map_pretty(|x, w| write!(w, "{}: {}", x.name.pretty(fmt), x.val.pretty(fmt)), ", "))
        }
        ast::TypeKind::Path(x, None)     => write!(w, "{}", x.pretty(fmt)),
        ast::TypeKind::Path(x, Some(ts)) => write!(w, "{}[{}]", x.pretty(fmt), ts.all_pretty(", ", fmt)),
        ast::TypeKind::Array(None, sh)      => write!(w, "[{}]", sh.pretty(fmt)),
        ast::TypeKind::Array(Some(ty), sh)  => write!(w, "[{}; {}]", ty.pretty(fmt), sh.pretty(fmt)),
        ast::TypeKind::Stream(ty)           => write!(w, "~{}", ty.pretty(fmt)),
        ast::TypeKind::Tuple(tys)           => write!(w, "({})", tys.iter().all_pretty(", ", fmt)),
        ast::TypeKind::Fun(args, ty)        => write!(w, "fun({}): {}", args.all_pretty(", ", fmt), ty.pretty(fmt)),
        ast::TypeKind::By(ty0, ty1)         => write!(w, "{} by {}", ty0.pretty(fmt), ty1.pretty(fmt)),
        ast::TypeKind::Err                  => write!(w, "☇"),
    },
    ast::NameId => write!(w, "{}", fmt.names.resolve(node)),
    ast::Name => write!(w, "{}", fmt.names.resolve(node)),
    ast::Shape => write!(w, "{}", node.dims.iter().all_pretty(", ", fmt)),
    ast::Dim => match &node.kind {
        ast::DimKind::Var(_)       => write!(w, "?"),
        ast::DimKind::Val(v)       => write!(w, "{}", v),
        ast::DimKind::Op(l, op, r) => write!(w, "{}{}{}", l.pretty(fmt), op.pretty(fmt), r.pretty(fmt)),
        ast::DimKind::Err          => write!(w, "☇"),
    },
    ast::DimOp => match &node.kind {
        ast::DimOpKind::Add => write!(w, "+"),
        ast::DimOpKind::Sub => write!(w, "-"),
        ast::DimOpKind::Mul => write!(w, "*"),
        ast::DimOpKind::Div => write!(w, "/"),
    },
    ast::Pat => {
        let kind = fmt.ast.pats.resolve(node);
        if fmt.mode.verbosity >= Verbosity::Debug {
            write!(w, "({})", kind.pretty(fmt))?;
        } else {
            write!(w, "{}", kind.pretty(fmt))?;
        }
    },
    ast::PatKind => match &node {
        ast::PatKind::Or(p0, p1)    => write!(w, "{} | {}", p0.pretty(fmt), p1.pretty(fmt)),
        ast::PatKind::By(p0, p1)    => write!(w, "{} by {}", p0.pretty(fmt), p1.pretty(fmt)),
        ast::PatKind::Const(l)      => write!(w, "{}", l.pretty(fmt)),
        ast::PatKind::Var(x)      => write!(w, "{}", x.pretty(fmt)),
        ast::PatKind::Tuple(ps)     => write!(w, "({})", ps.all_pretty(", ", fmt)),
        ast::PatKind::Struct(fs)    => write!(w, "{{ {fs} }}",
            fs = fs.map_pretty(|x, w| if let Some(p) = &x.val {
                        write!(w, "{}: {}", x.name.pretty(fmt), p.pretty(fmt))
                    } else {
                        write!(w, "{}", x.name.pretty(fmt))
                    }, ", ")
        ),
        ast::PatKind::Ignore        => write!(w, "_"),
        ast::PatKind::Variant(x, p) => write!(w, "{}({})", x.pretty(fmt), p.pretty(fmt)),
        ast::PatKind::Err           => write!(w, "☇"),
    },
    Box<ast::Dim> => write!(w, "{}", node.as_ref().pretty(fmt)),
}
