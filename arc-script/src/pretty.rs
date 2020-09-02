use crate::{ast::*, info::Info, utils::Printer};
use BinOpKind::*;
use DimKind::*;
use DimOpKind::*;
use ExprKind::*;
use LitKind::*;
use PatternKind::*;
use ScalarKind::*;
use TypeKind::*;
use UnOpKind::*;

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
                .map(|fun| fun.pretty(pr))
                .collect::<Vec<String>>()
                .join("\n"),
            body = self.ast.body.pretty(pr),
            s = pr.indent()
        )
    }
}

impl Pretty for FunDef {
    fn pretty(&self, pr: &Printer) -> String {
        format!(
            "{s0}fun {id}({params}){s1}{body}{s0}end{s0}",
            id = self.id.pretty(pr),
            params = self
                .params
                .iter()
                .map(|param| format!(
                    "{}: {}",
                    param.pretty(pr),
                    pr.info.table.get_decl(param).tv.pretty(pr)
                ))
                .collect::<Vec<_>>()
                .join(", "),
            body = self.body.pretty(&pr.tab()),
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
                "if {c}{s0}then{s1}{t}{s0}else{s1}{e}",
                c = c.pretty(&pr.tab()),
                t = t.pretty(&pr.tab().tab()),
                e = e.pretty(&pr.tab().tab()),
                s0 = pr.tab().indent(),
                s1 = pr.tab().tab().indent(),
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
            Let(id, e, b) => format!(
                "{id}: {ty} = {e}{s}{b}",
                id = id.pretty(pr),
                ty = pr.info.table.get_decl(id).tv.pretty(pr),
                e = e.pretty(pr),
                b = b.pretty(pr),
                s = pr.indent(),
            ),
            Closure(..) => todo!(),
            Lit(lit) => lit.pretty(pr),
            Var(id) => id.pretty(pr),
            BinOp(l, op, r) => format!(
                "{l} {op} {r}",
                l = l.pretty(pr),
                op = op.pretty(pr),
                r = r.pretty(pr)
            ),
            UnOp(op, e) => (op, e.as_ref()).pretty(pr),
            ConsArray(args) => format!("[{args}]", args = args.pretty(pr)),
            ConsStruct(fields) => format!("{{ {fields} }}", fields = fields.pretty(pr)),
            ConsTuple(args) => format!("({args})", args = args.pretty(pr)),
            FunCall(id, args) => {
                format!("{id}({args})", id = id.pretty(pr), args = args.pretty(pr))
            }
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
    fn pretty(&self, _: &Printer) -> String {
        match self {
            LitI8(l) => format!("{}i8", l.to_string()),
            LitI16(l) => format!("{}i16", l.to_string()),
            LitI32(l) => l.to_string(),
            LitI64(l) => format!("{}i64", l.to_string()),
            LitF32(l) => format!("{}f32", l.to_string()),
            LitF64(l) => l.to_string(),
            LitBool(l) => l.to_string(),
            LitTime(l) => l.to_string(),
            LitErr => "☇".to_string(),
        }
    }
}

impl<'i> Pretty for (&'i UnOpKind, &'i Expr) {
    fn pretty(&self, pr: &Printer) -> String {
        let (op, e) = self;
        match op {
            Not => format!("!{}", e.pretty(pr)),
            Cast(ty) => format!("{}:{}", e.pretty(pr), ty.pretty(pr)),
            MethodCall(id, args) => format!(
                "{e}.{id}({args})",
                e = e.pretty(pr),
                id = id.pretty(pr),
                args = args.pretty(pr)
            ),
            Project(idx) => format!("{}.{}", e.pretty(pr), idx.pretty(pr)),
            Access(_) => todo!(),
            UnOpErr => format!("☇{}", e.pretty(pr)),
        }
    }
}

impl Pretty for BinOpKind {
    fn pretty(&self, _: &Printer) -> String {
        match self {
            Add => "+".to_string(),
            Sub => "-".to_string(),
            Mul => "*".to_string(),
            Div => "/".to_string(),
            Eq => "==".to_string(),
            Neq => "!=".to_string(),
            Gt => ">".to_string(),
            Lt => "<".to_string(),
            Geq => ">=".to_string(),
            Leq => "<=".to_string(),
            BinOpErr => "☇".to_string(),
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
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            Scalar(I8) => "i8".to_string(),
            Scalar(I16) => "i16".to_string(),
            Scalar(I32) => "i32".to_string(),
            Scalar(I64) => "i64".to_string(),
            Scalar(F32) => "f32".to_string(),
            Scalar(F64) => "f64".to_string(),
            Scalar(Bool) => "bool".to_string(),
            Scalar(Null) => "null".to_string(),
            Scalar(Str) => "str".to_string(),
            Struct(fields) => format!("{{ {fields} }}", fields = fields.pretty(pr),),
            Array(ty, shape) => format!(
                "[{ty}; {shape}]",
                ty = ty.pretty(pr),
                shape = shape.pretty(pr)
            ),
            Tuple(tys) => format!("({})", tys.pretty(pr)),
            Optional(ty) => format!("{}?", ty.pretty(pr)),
            Fun(args, ty) => format!("({}) -> {}", args.pretty(pr), ty.pretty(pr)),
            Unknown => "?".to_string(),
            TypeErr => "☇".to_string(),
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
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            DimVar(_) => "?".to_owned(),
            DimVal(v) => format!("{}", v),
            DimOp(l, op, r) => format!("{}{}{}", l.pretty(pr), op.pretty(pr), r.pretty(pr)),
            DimErr => "☇".to_string(),
        }
    }
}

impl Pretty for DimOpKind {
    fn pretty(&self, _: &Printer) -> String {
        match self {
            DimAdd => "+".to_owned(),
            DimSub => "-".to_owned(),
            DimMul => "*".to_owned(),
            DimDiv => "/".to_owned(),
        }
    }
}

impl Pretty for Pattern {
    fn pretty(&self, pr: &Printer) -> String {
        match &self.kind {
            Regex(s) => format!(r#"r"{}""#, s.clone()),
            Or(l, r) => format!("{} | {}", l.pretty(pr), r.pretty(pr)),
            Val(l) => l.pretty(pr),
            Bind(x) => x.pretty(pr),
            DeconsTuple(vs) => format!("({})", vs.pretty(pr)),
            Wildcard => "_".to_owned(),
            PatternErr => "☇".to_string(),
        }
    }
}
