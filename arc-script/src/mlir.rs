use crate::{ast::*, info::Info, typer::*, utils::Printer};
use std::cell::RefMut;
use BinOpKind::*;
use ExprKind::*;
use LitKind::*;
use ScalarKind::*;
use TypeKind::*;

impl Script<'_> {
    pub fn mlir(&self) -> String {
        "todo".to_owned()
        //         self.ast.fundefs.
    }
}

impl Expr {
    pub fn mlir(&self, info: &Info) -> String {
        let pr = Printer {
            info,
            tabs: 2,
            verbose: false,
        };
        format!(
            "{s}{module}\n",
            module = self.to_module(&pr.tab()),
            s = pr.indent()
        )
    }

    pub fn to_module(&self, pr: &Printer) -> String {
        let main = format!(
            r#"{s}"std.func"() ({{{region}}}) {{ sym_name = "main", type = () -> {ty} }}"#,
            region = self.to_region("std.return", &pr.tab()),
            ty = self.tv.to_ty(pr),
            s = pr.indent(),
        );
        format!(
            r#""module" ({{{main}{s0}"module_terminator"() : () -> (){s1}}}) : () -> ()"#,
            main = main,
            s0 = pr.indent(),
            s1 = pr.untab().indent()
        )
    }

    fn to_region(&self, terminator: &str, pr: &Printer) -> String {
        match &self.kind {
            Let(id, v) => format!(
                "{s}{var} = {op}",
                var = id.to_var(),
                op = v.to_op(pr),
                s = pr.indent(),
            ),
            BinOp(lhs, Seq, rhs) => format!(
                "{s}{lhs}{rhs}",
                lhs = lhs.to_op(pr),
                rhs = rhs.to_region(terminator, pr),
                s = pr.indent(),
            ),
            Var(_) => format!(
                r#"{s0}"{t}"({var}) : ({ty}) -> (){s1}"#,
                t = terminator,
                var = self.to_var(),
                ty = self.tv.to_ty(pr),
                s0 = pr.indent(),
                s1 = pr.untab().indent(),
            ),
            _ => unreachable!(),
        }
    }

    fn to_op(&self, pr: &Printer) -> String {
        match &self.kind {
            Lit(kind) => {
                let lit = kind.to_lit();
                let ty = self.tv.to_ty(pr);
                format!(
                    r#""std.constant"() {{ value = {lit} : {ty} }}: () -> {ty}"#,
                    lit = lit,
                    ty = ty,
                )
            }
            BinOp(l, op, r) => {
                let l = l.to_var();
                let r = r.to_var();
                let ty = pr.info.typer.borrow_mut().lookup(self.tv);
                match (op, ty.kind) {
                    (Add, Scalar(I32)) => {
                        format!(r#""std.addi"({}, {}) : (i32, i32) -> i32"#, l, r)
                    }
                    (Add, Scalar(F32)) => {
                        format!(r#""std.addf"({}, {}) : (f32, f32) -> f32"#, l, r)
                    }
                    _ => todo!(),
                }
            }
            If(c, t, e) => format!(
                r#""arc.if"({var}) ({{{t}}},{{{e}}}) : ({arg_ty}) -> {out_ty}"#,
                var = c.to_var(),
                t = t.to_region("arc.yield", &pr.tab()),
                e = e.to_region("arc.yield", &pr.tab()),
                arg_ty = c.tv.to_ty(pr),
                out_ty = e.tv.to_ty(pr),
            ),
            UnOp(..) => todo!(),
            ConsArray(..) => todo!(),
            ConsStruct(..) => todo!(),
            ConsTuple(..) => todo!(),
            Closure(..) => todo!(),
            Let(..) => panic!("[ICE] Attempted to generate MLIR SSA of Let"),
            Match(..) => panic!("[ICE] Attempted to generate MLIR SSA of Match"),
            Var(_) => panic!("[ICE] Attempted to generate MLIR SSA of Var"),
            Sink(_) => todo!(),
            Source(_) => todo!(),
            Loop(_, _) => todo!(),
            ExprErr => "<ERROR>".to_owned(),
        }
    }

    fn to_var(&self) -> String {
        match &self.kind {
            Var(id) => id.to_var(),
            _ => unreachable!(),
        }
    }
}

impl Ident {
    fn to_var(&self) -> String {
        format!("%x_{}", self.0)
    }
}

impl TypeVar {
    fn to_ty(self, pr: &Printer) -> String {
        let typer = pr.info.typer.borrow_mut();
        self.to_ty_rec(typer)
    }
    fn to_ty_rec(self, mut typer: RefMut<Typer>) -> String {
        match typer.lookup(self).kind {
            Scalar(I8) => "i8".to_owned(),
            Scalar(I16) => "i16".to_owned(),
            Scalar(I32) => "i32".to_owned(),
            Scalar(I64) => "i64".to_owned(),
            Scalar(F32) => "f32".to_owned(),
            Scalar(F64) => "f64".to_owned(),
            Scalar(Bool) => "i1".to_owned(),
            Scalar(Null) => todo!(),
            Scalar(Str) => todo!(),
            Scalar(Unit) => todo!(),
            Struct(_) => todo!(),
            Array(_, _) => todo!(),
            Stream(_) => todo!(),
            Tuple(_) => todo!(),
            Optional(_) => todo!(),
            Fun(_, _) => todo!(),
            Unknown => "<UNKNOWN>".to_string(),
            TypeErr => "<ERROR>".to_string(),
        }
    }
}

impl LitKind {
    fn to_lit(&self) -> String {
        match self {
            LitI8(l) => l.to_string(),
            LitI16(l) => l.to_string(),
            LitI32(l) => l.to_string(),
            LitI64(l) => l.to_string(),
            LitF32(l) => l.to_string(),
            LitF64(l) => l.to_string(),
            LitBool(l) => l.to_string(),
            LitTime(_) => todo!(),
            LitUnit => todo!(),
            LitErr => "<ERROR>".to_string(),
        }
    }
}
