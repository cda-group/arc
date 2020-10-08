use crate::{info::Info, prelude::*, utils::Printer};

pub trait Pretty {
    fn pretty(&self, pr: &Printer) -> String;
    fn brief(&self, info: &Info) -> String {
        let pr = Printer {
            tabs: 0,
            verbose: false,
            info,
        };
        self.pretty(&pr)
    }
    fn code(&self, verbose: bool, info: &Info) -> String {
        let pr = Printer {
            tabs: 0,
            verbose,
            info,
        };
        self.pretty(&pr)
    }
}

impl Pretty for Script<'_> {
    fn pretty(&self, pr: &Printer) -> String {
        format!(
            "{funs}{s}{body}",
            funs = self
                .ast
                .fundefs
                .iter()
                .map(|(id, fun)| (id, fun).pretty(pr))
                .collect::<Vec<String>>()
                .join("\n"),
            body = self.ast.body.pretty(pr),
            s = pr.indent()
        )
    }
}

impl Pretty for (&Ident, &FunDef) {
    fn pretty(&self, pr: &Printer) -> String {
        let (id, fundef) = self;
        format!(
            "{s0}fun {id}({params}) {{{s1}{body}{s0}}}",
            id = id.pretty(pr),
            params = fundef
                .params
                .iter()
                .map(|param| format!(
                    "{}: {}",
                    param.pretty(pr),
                    pr.info.table.get_decl(param).tv.pretty(pr)
                ))
                .collect::<Vec<_>>()
                .join(", "),
            body = fundef.body.pretty(&pr.tab()),
            s0 = pr.indent(),
            s1 = pr.tab().indent(),
        )
    }
}

impl<A: Pretty> Pretty for Vec<A> {
    fn pretty(&self, pr: &Printer) -> String {
        self.iter()
            .map(|e| e.pretty(pr))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

impl<A: Pretty, B: Pretty> Pretty for (A, B) {
    fn pretty(&self, pr: &Printer) -> String {
        format!("{}: {}", self.0.pretty(pr), self.1.pretty(pr))
    }
}

impl Pretty for Expr {
    #[allow(clippy::many_single_char_names)]
    fn pretty(&self, pr: &Printer) -> String {
        let expr = match &self.kind {
            If(c, t, e) => format!(
                "if {c} {{{s1}{t}{s0}}} else {{{s1}{e}{s0}}}",
                c = c.pretty(&pr),
                t = t.pretty(&pr.tab()),
                e = e.pretty(&pr.tab()),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            For(p, e, c, b) => format!(
                "for {p} in {e} {c} {{{s1}{b}{s0}}}",
                p = p.pretty(&pr),
                e = e.pretty(&pr),
                c = if let Some(c) = c {
                    format!("where {c}", c = c.pretty(&pr))
                } else {
                    format!("")
                },
                b = b.pretty(&pr),
                s0 = pr.indent(),
                s1 = pr.tab().indent()
            ),
            Match(e, clauses) => format!(
                "match {e} {{{clause}{s0}}}{s1}",
                e = e.pretty(&pr.tab()),
                clause = clauses
                    .iter()
                    .map(|(p, e)| format!(
                        "{s2}{} => {}",
                        p.pretty(&pr.tab().tab()),
                        e.pretty(&pr.tab().tab()),
                        s2 = pr.indent()
                    ))
                    .collect::<Vec<String>>()
                    .join(","),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            Let(id, e) => format!(
                "let {id}: {ty} = {e}",
                id = id.pretty(pr),
                ty = pr.info.table.get_decl(id).tv.pretty(pr),
                e = e.pretty(pr),
            ),
            Closure(params, body) => format!(
                "|{params}| {{{s1}{body}{s0}}}",
                params = params
                    .iter()
                    .map(|id| format!(
                        "{id}:{ty}",
                        id = id.pretty(pr),
                        ty = pr.info.table.get_decl(id).tv.pretty(pr)
                    ))
                    .collect::<Vec<String>>()
                    .join(", "),
                body = body.pretty(&pr.tab()),
                s0 = pr.indent(),
                s1 = pr.tab().indent(),
            ),
            Lit(lit) => lit.pretty(pr),
            Var(id) => id.pretty(pr),
            BinOp(l, Seq, r) => format!(
                "{l};{s0}{r}",
                l = l.pretty(pr),
                r = r.pretty(pr),
                s0 = pr.indent(),
            ),
            BinOp(l, op, r) => format!(
                "{l} {op} {r}",
                l = l.pretty(pr),
                op = op.pretty(&pr.tab()),
                r = r.pretty(pr)
            ),
            UnOp(op, e) => (op, e.as_ref()).pretty(pr),
            ConsArray(args) => format!("[{args}]", args = args.pretty(pr)),
            ConsStruct(fields) => format!("{{ {fields} }}", fields = fields.pretty(pr)),
            ConsEnum(variants) => format!("{{ {variants} }}", variants = variants.pretty(pr)),
            ConsTuple(args) => format!("({args})", args = args.pretty(pr)),
            Sink(id) => format!("sink::{id}", id = id.pretty(pr)),
            Source(id) => format!("source::{id}", id = id.pretty(pr)),
            Loop(cond, body) => format!(
                "loop {cond} {{{s1}{body}}}",
                cond = cond.pretty(pr),
                body = body.pretty(pr),
                s1 = pr.tab().indent(),
            ),
            ExprErr => "☇".to_string(),
        };
        if pr.verbose {
            format!("({expr}):{ty}", expr = expr, ty = self.tv.pretty(pr))
        } else {
            expr
        }
    }
}

impl Pretty for LitKind {
    #[rustfmt::skip]
    fn pretty(&self, _: &Printer) -> String {
        match self {
            LitI8(l)   => format!("{}i8", l),
            LitI16(l)  => format!("{}i16", l),
            LitI32(l)  => format!("{}", l),
            LitI64(l)  => format!("{}i64", l),
            LitF32(l)  => format!("{}f32", l),
            LitF64(l)  => format!("{}", l),
            LitBool(l) => format!("{}", l),
            LitTime(l) => format!("{}", l),
            LitUnit    => format!("()"),
            LitErr     => "☇".to_string(),
        }
    }
}

impl Pretty for (&UnOpKind, &Expr) {
    #[rustfmt::skip]
    fn pretty(&self, pr: &Printer) -> String {
        let (op, e) = self;
        match op {
            Not          => format!("!{}", e.pretty(pr)),
            Neg          => format!("-{}", e.pretty(pr)),
            Cast(ty)     => format!("{}:{}", e.pretty(pr), ty.pretty(pr)),
            Project(idx) => format!("{}.{}", e.pretty(pr), idx.pretty(pr)),
            Access(_)    => todo!(),
            Call(args)   => format!("{e}({args})", e = e.pretty(pr), args = args.pretty(pr)),
            UnOpErr      => format!("☇{}", e.pretty(pr)),
        }
    }
}

impl Pretty for BinOpKind {
    #[rustfmt::skip]
    fn pretty(&self, _: &Printer) -> String {
        match self {
            Add      => format!("+"),
            Sub      => format!("-"),
            Mul      => format!("*"),
            Div      => format!("/"),
            Pow      => format!("**"),
            Equ      => format!("=="),
            Neq      => format!("!="),
            Gt       => format!(">"),
            Lt       => format!("<"),
            Geq      => format!(">="),
            Leq      => format!("<="),
            Or       => format!("||"),
            And      => format!("&&"),
            Pipe     => format!("|>"),
            Seq      => format!(";"),
            BinOpErr => format!("☇"),
        }
    }
}

impl Pretty for TypeVar {
    fn pretty(&self, pr: &Printer) -> String {
        let ty = { pr.info.typer.borrow_mut().lookup(*self) };
        ty.pretty(pr)
    }
}

impl Pretty for Type {
    #[rustfmt::skip]
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            Scalar(I8)       => format!("i8"),
            Scalar(I16)      => format!("i16"),
            Scalar(I32)      => format!("i32"),
            Scalar(I64)      => format!("i64"),
            Scalar(F32)      => format!("f32"),
            Scalar(F64)      => format!("f64"),
            Scalar(Bool)     => format!("bool"),
            Scalar(Null)     => format!("null"),
            Scalar(Str)      => format!("str"),
            Scalar(Unit)     => format!("()"),
            Struct(fields)   => format!("{{ {fields} }}", fields = fields.pretty(pr),),
            Enum(variants)   => format!("{{ {variants} }}", variants = variants.pretty(pr),),
            Array(ty, shape) => format!(
                "[{ty}; {shape}]",
                ty = ty.pretty(pr),
                shape = shape.pretty(pr)
            ),
            Stream(ty)       => format!("Stream[{}]", ty.pretty(pr)),
            Map(kty, vty)    => format!("Map[{},{}]", kty.pretty(pr), vty.pretty(pr)),
            Set(ty)          => format!("Set[{}]", ty.pretty(pr)),
            Vector(ty)       => format!("Vec[{}]", ty.pretty(pr)),
            Tuple(tys)       => format!("({})", tys.pretty(pr)),
            Optional(ty)     => format!("{}?", ty.pretty(pr)),
            Fun(args, ty)    => format!("({}) -> {}", args.pretty(pr), ty.pretty(pr)),
            Unknown          => format!("?"),
            TypeErr          => format!("☇"),
        }
    }
}

impl Pretty for Index {
    fn pretty(&self, _: &Printer) -> String {
        format!("{}", self.0)
    }
}

impl Pretty for Ident {
    fn pretty(&self, pr: &Printer) -> String {
        pr.info.table.get_decl_name(self).to_string()
    }
}

impl Pretty for Field {
    fn pretty(&self, pr: &Printer) -> String {
        pr.info.table.resolve(&self.key).to_string()
    }
}

impl Pretty for Variant {
    fn pretty(&self, pr: &Printer) -> String {
        pr.info.table.resolve(&self.key).to_string()
    }
}

impl Pretty for SymbolKey {
    fn pretty(&self, pr: &Printer) -> String {
        pr.info.table.resolve(self).to_string()
    }
}

impl Pretty for Shape {
    fn pretty(&self, pr: &Printer) -> String {
        self.dims
            .iter()
            .map(|dim| dim.pretty(pr))
            .collect::<Vec<String>>()
            .join(",")
    }
}

impl Pretty for Dim {
    #[rustfmt::skip]
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            DimVar(_)       => format!("?"),
            DimVal(v)       => format!("{}", v),
            DimOp(l, op, r) => format!("{}{}{}", l.pretty(pr), op.pretty(pr), r.pretty(pr)),
            DimErr          => format!("☇"),
        }
    }
}

impl Pretty for DimOpKind {
    fn pretty(&self, _: &Printer) -> String {
        match self {
            DimAdd => format!("+"),
            DimSub => format!("-"),
            DimMul => format!("*"),
            DimDiv => format!("/"),
        }
    }
}

impl Pretty for Pat {
    #[rustfmt::skip]
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            PatRegex(s)   => format!(r#"r"{}""#, s.clone()),
            PatOr(l, r)   => format!("{} | {}", l.pretty(pr), r.pretty(pr)),
            PatVal(l)     => l.pretty(pr),
            PatVar(x)     => x.pretty(pr),
            PatTuple(vs)  => format!("({})", vs.pretty(pr)),
            PatStruct(vs) => format!("{{ {} }}", vs.pretty(pr)),
            PatIgnore     => format!("_"),
            PatErr        => format!("☇"),
        }
    }
}

impl<V> Pretty for VecMap<Field, V>
where
    V: Pretty,
{
    fn pretty(&self, pr: &Printer) -> String {
        self.iter()
            .map(|(k, v)| format!("{}:{}", k.pretty(pr), v.pretty(pr)))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

impl<V> Pretty for VecMap<Variant, V>
where
    V: Pretty,
{
    fn pretty(&self, pr: &Printer) -> String {
        self.iter()
            .map(|(k, v)| format!("{} of {}", k.pretty(pr), v.pretty(pr)))
            .collect::<Vec<String>>()
            .join(", ")
    }
}
