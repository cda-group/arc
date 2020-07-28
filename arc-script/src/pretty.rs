use {
    crate::{ast::*, utils::indent},
    std::fmt::*,
    BinOpKind::*,
    DimKind::*,
    ExprKind::*,
    LitKind::*,
    PatternKind::*,
    ScalarKind::*,
    ShapeKind::*,
    TypeKind::*,
    UnOpKind::*,
};

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{}", self.pretty(0, false)) }
}

impl Display for TypeKind {
    fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{}", self.pretty(0, false)) }
}

impl Display for Uid {
    fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{}", self.0) }
}

pub trait Pretty {
    /// `i` is the indentation level
    /// `v` is the verbosity
    fn pretty(&self, i: u32, v: bool) -> String;
    fn code(&self, v: bool) -> String {
        let i = 2;
        format!("{}\n", self.pretty(i, v))
    }
}

impl Pretty for Script {
    fn pretty(&self, i: u32, v: bool) -> String {
        format!(
            "{funs}{s}{body}",
            funs = self
                .funs
                .iter()
                .map(|fun| fun.pretty(i, v))
                .collect::<Vec<String>>()
                .join("\n"),
            body = self.body.pretty(i, v),
            s = indent(i)
        )
    }
}

impl Pretty for FunDef {
    fn pretty(&self, i: u32, v: bool) -> String {
        format!(
            "{s0}fun {id}({params}){s1}{body}{s0}end{s0}",
            id = self.id.pretty(i, v),
            params = self.params.pretty(i, v),
            body = self.body.pretty(i + 1, v),
            s0 = indent(i),
            s1 = indent(i + 1),
        )
    }
}

impl<A: Pretty> Pretty for Vec<A> {
    fn pretty(&self, i: u32, v: bool) -> String {
        self.iter()
            .map(|e| format!("{}", e.pretty(i, v)))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

impl<A: Pretty, B: Pretty> Pretty for (A, B) {
    fn pretty(&self, i: u32, v: bool) -> String {
        format!("{}: {}", self.0.pretty(i, v), self.1.pretty(i, v))
    }
}

impl Pretty for Expr {
    fn pretty(&self, i: u32, v: bool) -> String {
        let expr = match &self.kind {
            If(c, t, e) => format!(
                "if {c} {s0}then {s1}{t} {s0}else {s1}{e}",
                c = c.pretty(i + 1, v),
                t = t.pretty(i + 2, v),
                e = e.pretty(i + 2, v),
                s0 = indent(i + 1),
                s1 = indent(i + 2),
            ),
            Match(e, clauses) => format!(
                "match {e} {{{clause}{s0}}}{s1}",
                e = e.pretty(i + 1, v),
                clause = clauses
                    .iter()
                    .map(|(p, e)| format!(
                        "{s2}{} => {}",
                        p.pretty(i + 2, v),
                        e.pretty(i + 2, v),
                        s2 = indent(i + 2)
                    ))
                    .collect::<Vec<String>>()
                    .join(","),
                s0 = indent(i),
                s1 = indent(i + 1),
            ),
            Let(id, ty, e, b) => format!(
                "{id}: {ty} = {e}{s}{b}",
                id = id.pretty(i, v),
                ty = ty.pretty(i, v),
                e = e.pretty(i, v),
                b = b.pretty(i, v),
                s = indent(i),
            ),
            Lit(lit) => lit.pretty(i, v),
            Var(id) => id.pretty(i, v),
            BinOp(l, op, r) => format!(
                "{l} {op} {r}",
                l = l.pretty(i, v),
                op = op.pretty(i, v),
                r = r.pretty(i, v)
            ),
            UnOp(op, e) => (op, e.as_ref()).pretty(i, v),
            ConsArray(args) => format!("[{args}]", args = args.pretty(i, v)),
            ConsStruct(fields) => format!("{{ {fields} }}", fields = fields.pretty(i, v)),
            ConsTuple(args) => format!("({args})", args = args.pretty(i, v)),
            FunCall(id, args) => format!(
                "{id}({args})",
                id = id.pretty(i, v),
                args = args.pretty(i, v)
            ),
            ExprErr => "☇".to_string(),
        };
        if v {
            format!("({expr}):{ty}", expr = expr, ty = self.ty.pretty(i, v))
        } else {
            expr
        }
    }
}

impl Pretty for LitKind {
    fn pretty(&self, _: u32, _: bool) -> String {
        match self {
            LitI32(l) => l.to_string(),
            LitI64(l) => format!("{}i64", l.to_string()),
            LitF32(l) => format!("{}f32", l.to_string()),
            LitF64(l) => l.to_string(),
            LitBool(l) => l.to_string(),
            LitErr => "☇".to_string(),
        }
    }
}

impl<'i> Pretty for (&'i UnOpKind, &'i Expr) {
    fn pretty(&self, i: u32, v: bool) -> String {
        let (op, e) = self;
        match op {
            Not => format!("!{}", e.pretty(i, v)),
            Cast(ty) => format!("{}:{}", e.pretty(i, v), ty.pretty(i, v)),
            MethodCall(id, args) => format!(
                "{e}.{id}({args})",
                e = e.pretty(i, v),
                id = id.pretty(i, v),
                args = args.pretty(i, v)
            ),
            Project(idx) => format!("{}.{}", e.pretty(i, v), idx.pretty(i, v)),
            Access(_) => todo!(),
            UnOpErr => format!("☇{}", e.pretty(i, v)),
        }
    }
}

impl Pretty for BinOpKind {
    fn pretty(&self, _: u32, _: bool) -> String {
        match self {
            Add => "+".to_string(),
            Sub => "-".to_string(),
            Mul => "*".to_string(),
            Div => "/".to_string(),
            Eq => "==".to_string(),
            BinOpErr => "☇".to_string(),
        }
    }
}

impl Pretty for Type {
    fn pretty(&self, i: u32, v: bool) -> String { self.kind.pretty(i, v) }
}

impl Pretty for TypeKind {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self {
            Scalar(I8) => "i8".to_string(),
            Scalar(I16) => "i16".to_string(),
            Scalar(I32) => "i32".to_string(),
            Scalar(I64) => "i64".to_string(),
            Scalar(F32) => "f32".to_string(),
            Scalar(F64) => "f64".to_string(),
            Scalar(Bool) => "bool".to_string(),
            Scalar(Null) => "null".to_string(),
            Scalar(Str) => "str".to_string(),
            Struct(fields) => format!("{{ {fields} }}", fields = fields.pretty(i, v),),
            Array(ty, shape) => format!(
                "[{ty}; {shape}]",
                ty = ty.pretty(i, v),
                shape = shape.pretty(i, v)
            ),
            Tuple(tys) => format!("({})", tys.pretty(i, v)),
            Optional(ty) => format!("{}?", ty.pretty(i, v)),
            Fun(_, _) => "".to_owned(),
            Unknown => "?".to_string(),
            TypeErr => "☇".to_string(),
        }
    }
}

impl Pretty for Index {
    fn pretty(&self, _: u32, _: bool) -> String { format!("{}", self.0) }
}

impl Pretty for Ident {
    fn pretty(&self, _: u32, _: bool) -> String {
        format!(
            "{name}{uid}",
            name = self.name,
            uid = self
                .uid
                .map(|uid| uid.to_string())
                .unwrap_or("?".to_string())
        )
    }
}

impl Pretty for Shape {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self.kind {
            Unranked => "*".to_owned(),
            Ranked(dims) => dims.pretty(i, v),
        }
    }
}

impl Pretty for Dim {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self.kind {
            Hole => "?".to_owned(),
            Symbolic(expr) => expr.pretty(i, v),
        }
    }
}

impl Pretty for Pattern {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self.kind {
            Regex(s) => format!(r#"r"{}""#, s.clone()),
            Or(l, r) => format!("{} | {}", l.pretty(i, v), r.pretty(i, v)),
            Val(l) => l.pretty(i, v),
            Bind(x) => x.pretty(i, v),
            DeconsTuple(vs) => format!("({})", vs.pretty(i, v)),
            Wildcard => "_".to_owned(),
            PatternErr => "☇".to_string(),
        }
    }
}
