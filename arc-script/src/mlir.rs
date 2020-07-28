use crate::{ast::*, utils::indent};
use BinOpKind::*;
use ExprKind::*;
use LitKind::*;
use ScalarKind::*;
use TypeKind::*;

impl Expr {
    pub fn mlir(&self) -> String {
        let i = 2;
        format!(
            "{s}{module}\n",
            module = self.to_module(i + 1),
            s = indent(i)
        )
    }

    pub fn to_module(&self, i: u32) -> String {
        let main = format!(
            r#"{s}"std.func"() ({{{region}}}) {{ sym_name = "main", type = () -> {ty} }}"#,
            region = self.to_region("std.return", i + 1),
            ty = self.ty.to_ty(),
            s = indent(i),
        );
        format!(
            r#""module" ({{{main}{s0}"module_terminator"() : () -> (){s1}}}) : () -> ()"#,
            main = main,
            s0 = indent(i),
            s1 = indent(i - 1)
        )
    }

    fn to_region(&self, terminator: &str, i: u32) -> String {
        match &self.kind {
            Let(id, _, v, b) => format!(
                "{s}{var} = {op}{next}",
                var = id.to_var(),
                op = v.to_op(i),
                next = b.to_region(terminator, i),
                s = indent(i),
            ),
            Var(_) => format!(
                r#"{s0}"{t}"({var}) : ({ty}) -> (){s1}"#,
                t = terminator,
                var = self.to_var(),
                ty = self.ty.to_ty(),
                s0 = indent(i),
                s1 = indent(i - 1),
            ),
            _ => unreachable!(),
        }
    }

    fn to_op(&self, i: u32) -> String {
        match &self.kind {
            Lit(kind) => {
                let lit = kind.to_lit();
                let ty = self.ty.to_ty();
                format!(
                    r#""std.constant"() {{ value = {lit} : {ty} }}: () -> {ty}"#,
                    lit = lit,
                    ty = ty,
                )
            }
            BinOp(l, op, r) => {
                let l = l.to_var();
                let r = r.to_var();
                match (op, &self.ty.kind) {
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
                t = t.to_region("arc.yield", i + 1),
                e = e.to_region("arc.yield", i + 1),
                arg_ty = c.ty.to_ty(),
                out_ty = e.ty.to_ty(),
            ),
            UnOp(..) => todo!(),
            ConsArray(..) => todo!(),
            ConsStruct(..) => todo!(),
            ConsTuple(..) => todo!(),
            FunCall(..) => todo!(),
            Let(..) => panic!("[ICE] Attempted to generate MLIR SSA of Let"),
            Match(..) => panic!("[ICE] Attempted to generate MLIR SSA of Match"),
            Var(_) => panic!("[ICE] Attempted to generate MLIR SSA of Var"),
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
        format!(
            "%x_{}",
            self.uid
                .expect("[ICE]: Attempted to generate MLIR variable without uid")
        )
    }
}

impl Type {
    fn to_ty(&self) -> String {
        match &self.kind {
            Scalar(I8) => "i8".to_owned(),
            Scalar(I16) => "i16".to_owned(),
            Scalar(I32) => "i32".to_owned(),
            Scalar(I64) => "i64".to_owned(),
            Scalar(F32) => "f32".to_owned(),
            Scalar(F64) => "f64".to_owned(),
            Scalar(Bool) => "i1".to_owned(),
            Scalar(Null) => todo!(),
            Scalar(Str) => todo!(),
            Struct(_) => todo!(),
            Array(_, _) => todo!(),
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
            LitI32(l) => l.to_string(),
            LitI64(l) => l.to_string(),
            LitF32(l) => l.to_string(),
            LitF64(l) => l.to_string(),
            LitBool(l) => l.to_string(),
            LitErr => "<ERROR>".to_string(),
        }
    }
}
