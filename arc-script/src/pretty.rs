use {
    crate::{ast::*, utils::indent},
    std::fmt::*,
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

impl Pretty for Fun {
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
            ExprKind::If(c, t, e) => format!(
                "if {c} {s0}then {s1}{t} {s0}else {s1}{e}",
                c = c.pretty(i + 1, v),
                t = t.pretty(i + 2, v),
                e = e.pretty(i + 2, v),
                s0 = indent(i + 1),
                s1 = indent(i + 2),
            ),
            ExprKind::Match(e, clauses) => format!(
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
            ExprKind::Let(id, ty, e, b) => format!(
                "{id}: {ty} = {e}{s}{b}",
                id = id.pretty(i, v),
                ty = ty.pretty(i, v),
                e = e.pretty(i, v),
                b = b.pretty(i, v),
                s = indent(i),
            ),
            ExprKind::Bif(_) => todo!(),
            ExprKind::Lit(lit) => lit.pretty(i, v),
            ExprKind::Var(id) => id.pretty(i, v),
            ExprKind::BinOp(l, op, r) => format!(
                "{l} {op} {r}",
                l = l.pretty(i, v),
                op = op.pretty(i, v),
                r = r.pretty(i, v)
            ),
            ExprKind::UnOp(op, e) => (op, e.as_ref()).pretty(i, v),
            ExprKind::Array(args) => format!("[{args}]", args = args.pretty(i, v)),
            ExprKind::Struct(fields) => format!("{{ {fields} }}", fields = fields.pretty(i, v),),
            ExprKind::Tuple(args) => format!("({args})", args = args.pretty(i, v)),
            ExprKind::Call(id, args) => format!(
                "{id}({args})",
                id = id.pretty(i, v),
                args = args.pretty(i, v)
            ),
            ExprKind::Error => "☇".to_string(),
        };
        if v {
            format!("({expr}):{ty}", expr = expr, ty = self.ty.pretty(i, v))
        } else {
            expr
        }
    }
}

impl Pretty for Lit {
    fn pretty(&self, _: u32, _: bool) -> String {
        match self {
            Lit::I32(l) => l.to_string(),
            Lit::I64(l) => format!("{}i64", l.to_string()),
            Lit::F32(l) => format!("{}f32", l.to_string()),
            Lit::F64(l) => l.to_string(),
            Lit::Bool(l) => l.to_string(),
        }
    }
}

impl<'i> Pretty for (&'i UnOp, &'i Expr) {
    fn pretty(&self, i: u32, v: bool) -> String {
        let (op, e) = self;
        match op {
            UnOp::Not => format!("!{}", e.pretty(i, v)),
            UnOp::Cast(ty) => format!("{}:{}", e.pretty(i, v), ty.pretty(i, v)),
            UnOp::Call(id, args) => format!(
                "{e}.{id}({args})",
                e = e.pretty(i, v),
                id = id.pretty(i, v),
                args = args.pretty(i, v)
            ),
            UnOp::Project(idx) => format!("{}.{}", e.pretty(i, v), idx.pretty(i, v)),
            UnOp::Access(_) => todo!(),
            UnOp::Error => format!("☇{}", e.pretty(i, v)),
        }
    }
}

impl Pretty for BinOp {
    fn pretty(&self, _: u32, _: bool) -> String {
        match self {
            BinOp::Add => "+".to_string(),
            BinOp::Sub => "-".to_string(),
            BinOp::Mul => "*".to_string(),
            BinOp::Div => "/".to_string(),
            BinOp::Eq => "==".to_string(),
            BinOp::Error => "☇".to_string(),
        }
    }
}

impl Pretty for Type {
    fn pretty(&self, i: u32, v: bool) -> String { self.kind.pretty(i, v) }
}

impl Pretty for TypeKind {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self {
            TypeKind::I8 => "i8".to_string(),
            TypeKind::I16 => "i16".to_string(),
            TypeKind::I32 => "i32".to_string(),
            TypeKind::I64 => "i64".to_string(),
            TypeKind::F32 => "f32".to_string(),
            TypeKind::F64 => "f64".to_string(),
            TypeKind::Bool => "bool".to_string(),
            TypeKind::Null => "null".to_string(),
            TypeKind::String => "str".to_string(),
            TypeKind::Struct(fields) => format!("{{ {fields} }}", fields = fields.pretty(i, v),),
            TypeKind::Array(ty, shape) => format!(
                "[{ty}; {shape}]",
                ty = ty.pretty(i, v),
                shape = shape.pretty(i, v)
            ),
            TypeKind::Tuple(tys) => format!("({})", tys.pretty(i, v)),
            TypeKind::Option(ty) => format!("{}?", ty.pretty(i, v)),
            TypeKind::Unknown => "?".to_string(),
            TypeKind::Error => "☇".to_string(),
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
            ShapeKind::Unranked => "*".to_owned(),
            ShapeKind::Ranked(dims) => dims.pretty(i, v),
        }
    }
}

impl Pretty for Dim {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self.kind {
            DimKind::Unknown => "?".to_owned(),
            DimKind::Expr(expr) => expr.pretty(i, v),
        }
    }
}

impl Pretty for Pattern {
    fn pretty(&self, i: u32, v: bool) -> String {
        match &self.kind {
            PatternKind::Regex(s) => format!(r#"r"{}""#, s.clone()),
            PatternKind::Error => "☇".to_string(),
            PatternKind::Or(l, r) => format!("{} | {}", l.pretty(i, v), r.pretty(i, v)),
            PatternKind::Lit(l) => l.pretty(i, v),
            PatternKind::Var(x) => x.pretty(i, v),
            PatternKind::Tuple(vs) => format!("({})", vs.pretty(i, v)),
            PatternKind::Wildcard => "_".to_owned(),
        }
    }
}
