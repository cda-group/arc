#![allow(clippy::useless_format)]
pub use crate::codegen::pretty_utils::*;
use crate::prelude::*;
use crate::printer::Printer;
use std::fmt::{self, Display, Formatter};

impl Script<'_> {
    pub fn code(&self, verbose: bool) -> String {
        let mut pr = Printer::from(&self.info);
        pr.verbose = verbose;
        self.pretty(&pr).to_string()
    }
}

impl Display for Pretty<'_, Script<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(script, pr) = self;
        write!(f, "{}", script.ast.fundefs.values().all_pretty("\n", pr))
    }
}

impl Display for Pretty<'_, FunDef> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(fundef, pr) = self;
        write!(
            f,
            "{s0}fun {id}({params}) {{{s1}{body}{s0}}}",
            id = fundef.id.pretty(pr),
            params = fundef.params.iter().map_pretty(
                |param, f| {
                    let tv = pr.info.table.get_decl(param).tv;
                    write!(f, "{}: {}", param.pretty(pr), tv.pretty(pr))
                },
                ", "
            ),
            body = fundef.body.pretty(&pr.tab()),
            s0 = pr.indent(),
            s1 = pr.tab().indent(),
        )
    }
}

impl Display for Pretty<'_, Expr> {
    #[allow(clippy::many_single_char_names)]
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(expr, pr) = self;
        if pr.verbose {
            write!(f, "(")?;
        }
        match &expr.kind {
            If(e0, e1, e2) => write!(
                f,
                "if {e0} {{{s1}{e1}{s0}}} else {{{s1}{e2}{s0}}}",
                e0 = e0.as_ref().pretty(&pr),
                e1 = e1.as_ref().pretty(&pr.tab()),
                e2 = e2.as_ref().pretty(&pr.tab()),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            For(p, e0, None, e2) => write!(
                f,
                "for {p} in {e0} {{{s1}{e2}{s0}}}",
                p = p.pretty(&pr),
                e0 = e0.as_ref().pretty(&pr),
                e2 = e2.as_ref().pretty(&pr),
                s0 = pr.indent(),
                s1 = pr.tab().indent()
            ),
            For(p, e0, Some(e1), e2) => write!(
                f,
                "for {p} in {e0} where {e1} {{{s1}{e2}{s0}}}",
                p = p.pretty(&pr),
                e0 = e0.as_ref().pretty(&pr),
                e1 = e1.as_ref().pretty(&pr),
                e2 = e2.as_ref().pretty(&pr),
                s0 = pr.indent(),
                s1 = pr.tab().indent()
            ),
            Match(e, cs) => write!(
                f,
                "match {e} {{{cs}{s0}}}{s1}",
                e = e.as_ref().pretty(&pr.tab()),
                cs = cs.map_pretty(
                    |(p, e), f| write!(f, "{} => {}", p.pretty(pr), e.pretty(pr)),
                    ", "
                ),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            Let(id, e0, e1) => write!(
                f,
                "let {id}: {ty} = {e0} in{s}{e1}",
                id = id.pretty(pr),
                ty = pr.info.table.get_decl(id).tv.pretty(pr),
                e0 = e0.as_ref().pretty(pr),
                e1 = e1.as_ref().pretty(pr),
                s = pr.indent()
            ),
            Closure(ps, e0) => write!(
                f,
                "|{ps}| {{{s1}{e0}{s0}}}",
                ps = ps.iter().map_pretty(
                    |x, f| {
                        let tv = pr.info.table.get_decl(x).tv;
                        write!(f, "{}: {}", x.pretty(pr), tv.pretty(pr))
                    },
                    ", "
                ),
                e0 = e0.as_ref().pretty(&pr.tab()),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            Lit(l) => write!(f, "{}", l.pretty(pr)),
            Var(x) => write!(f, "{}", x.pretty(pr)),
            BinOp(e0, Seq, e1) => write!(
                f,
                "{e0};{s0}{e1}",
                e0 = e0.as_ref().pretty(pr),
                e1 = e1.as_ref().pretty(pr),
                s0 = pr.indent(),
            ),
            BinOp(e0, op, e1) => write!(
                f,
                "{e0} {op} {e1}",
                e0 = e0.as_ref().pretty(pr),
                op = op.pretty(&pr.tab()),
                e1 = e1.as_ref().pretty(pr)
            ),
            UnOp(op, e0) => match op {
                Not        => write!(f, "not {}", e0.as_ref().pretty(pr)),
                Neg        => write!(f, "-{}", e0.as_ref().pretty(pr)),
                Cast(ty)   => write!(f, "{}:{}", e0.as_ref().pretty(pr), ty.pretty(pr)),
                Project(i) => write!(f, "{}.{}", e0.as_ref().pretty(pr), i.pretty(pr)),
                Access(_)  => todo!(),
                Call(es)   => write!(f, "{}({})", e0.as_ref().pretty(pr), es.all_pretty(", ", pr)),
                Emit       => write!(f, "emit {e0}", e0 = e0.as_ref().pretty(pr)),
                UnOpErr    => write!(f, "☇{}", e0.as_ref().pretty(pr)),
            },
            ConsArray(es) => write!(f, "[{es}]", es = es.all_pretty(", ", pr)),
            ConsStruct(_fs) => todo!(), //write!(f, "{{ {fields} }}", fields = pretty_fields(fields, pr)),
            ConsVariant(s, e0) => write!(
                f,
                "{{ {s}({e0}) }}",
                s = s.pretty(pr),
                e0 = e0.as_ref().pretty(pr)
            ),
            ConsTuple(es) => write!(f, "({es})", es = es.all_pretty(", ", pr)),
            Loop(e0, e1) => write!(
                f,
                "loop {e0} {{{s1}{e1}}}",
                e0 = e0.as_ref().pretty(pr),
                e1 = e1.as_ref().pretty(pr),
                s1 = pr.tab().indent(),
            ),
            ExprErr => write!(f, "☇"),
        }?;
        if pr.verbose {
            write!(f, "):{ty}", ty = expr.tv.pretty(pr))?
        };
        Ok(())
    }
}

impl Display for Pretty<'_, LitKind> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(lit, _) = self;
        match lit {
            LitI8(l)   => write!(f, "{}i8", l),
            LitI16(l)  => write!(f, "{}i16", l),
            LitI32(l)  => write!(f, "{}", l),
            LitI64(l)  => write!(f, "{}i64", l),
            LitF32(l)  => write!(f, "{}f32", l),
            LitF64(l)  => write!(f, "{}", l),
            LitBool(l) => write!(f, "{}", l),
            LitTime(l) => write!(f, "{}", l),
            LitUnit    => write!(f, "()"),
            LitErr     => write!(f, "☇"),
        }
    }
}

impl Display for Pretty<'_, BinOpKind> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _) = self;
        match op {
            Add      => write!(f, "+"),
            Sub      => write!(f, "-"),
            Mul      => write!(f, "*"),
            Div      => write!(f, "/"),
            Pow      => write!(f, "**"),
            Equ      => write!(f, "=="),
            Neq      => write!(f, "!="),
            Gt       => write!(f, ">"),
            Lt       => write!(f, "<"),
            Geq      => write!(f, ">="),
            Leq      => write!(f, "<="),
            Or       => write!(f, "or"),
            And      => write!(f, "and"),
            Pipe     => write!(f, "|>"),
            Seq      => write!(f, ";"),
            BinOpErr => write!(f, "☇"),
        }
    }
}

impl Display for Pretty<'_, TypeVar> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(tv, pr) = self;
        write!(f, "{}", pr.lookup(**tv).pretty(pr))
    }
}

impl Display for Pretty<'_, Type> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(ty, pr) = self;
        match &ty.kind {
            Scalar(kind) => match kind {
                I8   => write!(f, "i8"),
                I16  => write!(f, "i16"),
                I32  => write!(f, "i32"),
                I64  => write!(f, "i64"),
                F32  => write!(f, "f32"),
                F64  => write!(f, "f64"),
                Bool => write!(f, "bool"),
                Null => write!(f, "null"),
                Str  => write!(f, "str"),
                Unit => write!(f, "()"),
            }
            Struct(fs) => {
                write!(f, "{{ {fs} }}",
                    fs = fs.clone()
                           .into_inner()
                           .iter()
                           .map_pretty(|(k,v),f| write!(f, "{}: {}", k.pretty(pr), v.pretty(pr)), ","))
            }
            Enum(vs) =>{
                write!(f, "{{ {vs} }}",
                    vs = vs.clone()
                           .into_inner()
                           .iter()
                           .map_pretty(|(k,v),f| write!(f, "{}({})", k.pretty(pr), v.pretty(pr)), ","))
            }
            Nominal(x)       => write!(f, "{x}", x = x.pretty(pr)),
            Array(ty, shape) => write!(f,  "[{ty}; {shape}]", ty = ty.pretty(pr), shape = shape.pretty(pr)),
            Stream(ty)       => write!(f, "Stream[{}]", ty.pretty(pr)),
            Map(kty, vty)    => write!(f, "Map[{},{}]", kty.pretty(pr), vty.pretty(pr)),
            Set(ty)          => write!(f, "Set[{}]", ty.pretty(pr)),
            Vector(ty)       => write!(f, "Vec[{}]", ty.pretty(pr)),
            Tuple(tys)       => write!(f, "({})", tys.all_pretty(", ", pr)),
            Optional(ty)     => write!(f, "{}?", ty.pretty(pr)),
            Fun(args, ty)    => write!(f, "({}) -> {}", args.all_pretty(", ", pr), ty.pretty(pr)),
            Task(_)          => write!(f, ""),
            Unknown          => write!(f, "?"),
            TypeErr          => write!(f, "☇"),
        }
    }
}

impl Display for Pretty<'_, Index> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(Index(i), _) = self;
        write!(f, "{}", i)
    }
}

impl Display for Pretty<'_, Ident> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(id, pr) = self;
        write!(f, "{}", pr.info.table.get_decl_name(id))
    }
}

impl Display for Pretty<'_, Symbol> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(sym, pr) = self;
        write!(f, "{}", pr.info.table.resolve(sym))
    }
}

impl Display for Pretty<'_, Shape> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(shape, pr) = self;
        write!(f, "{}", shape.dims.iter().all_pretty(", ", pr))
    }
}

impl Display for Pretty<'_, Dim> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(dim, pr) = self;
        match &dim.kind {
            DimVar(_)       => write!(f, "?"),
            DimVal(v)       => write!(f, "{}", v),
            DimOp(l, op, r) => write!(f, "{}{}{}", l.as_ref().pretty(pr), op.pretty(pr), r.as_ref().pretty(pr)),
            DimErr          => write!(f, "☇"),
        }
    }
}

impl Display for Pretty<'_, DimOpKind> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(op, _pr) = self;
        match op {
            DimAdd => write!(f, "+"),
            DimSub => write!(f, "-"),
            DimMul => write!(f, "*"),
            DimDiv => write!(f, "/"),
        }
    }
}

impl Display for Pretty<'_, Pat> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Pretty(pat, pr) = self;
        match &pat.kind {
            PatRegex(s)   => write!(f, r#"r"{}""#, s.clone()),
            PatOr(p0, p1) => write!(f, "{} | {}", p0.as_ref().pretty(pr), p1.as_ref().pretty(pr)),
            PatVal(l)     => write!(f, "{}", l.pretty(pr)),
            PatVar(x)     => write!(f, "{}", x.pretty(pr)),
            PatTuple(ps)  => write!(f, "({})", ps.all_pretty(", ", pr)),
            PatStruct(ps) => write!(f, "{{ {} }}", ps.all_pretty(", ", pr)),
            PatIgnore     => write!(f, "_"),
            PatErr        => write!(f, "☇"),
        }
    }
}
