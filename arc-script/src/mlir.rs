use crate::{ast::*, utils::indent};

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
            ExprKind::Let(id, _, v, b) => format!(
                "{s}{var} = {op}{next}",
                var = id.to_var(),
                op = v.to_op(i),
                next = b.to_region(terminator, i),
                s = indent(i),
            ),
            ExprKind::Var(_) => format!(
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
            ExprKind::Lit(kind) => {
                let lit = kind.to_lit();
                let ty = self.ty.to_ty();
                format!(
                    r#""std.constant"() {{ value = {lit} : {ty} }}: () -> {ty}"#,
                    lit = lit,
                    ty = ty,
                )
            }
            ExprKind::Array(_) => todo!(),
            ExprKind::Tuple(_) => todo!(),
            ExprKind::Struct(_) => todo!(),
            ExprKind::BinOp(l, op, r) => {
                use BinOp::*;
                use TypeKind::*;
                let l = l.to_var();
                let r = r.to_var();
                match (op, &self.ty.kind) {
                    (Add, I32) => format!(r#""std.addi"({}, {}) : (i32, i32) -> i32"#, l, r),
                    (Add, F32) => format!(r#""std.addf"({}, {}) : (f32, f32) -> f32"#, l, r),
                    _ => todo!(),
                }
            }
            ExprKind::If(c, t, e) => format!(
                r#""arc.if"({var}) ({{{t}}},{{{e}}}) : ({arg_ty}) -> {out_ty}"#,
                var = c.to_var(),
                t = t.to_region("arc.yield", i + 1),
                e = e.to_region("arc.yield", i + 1),
                arg_ty = c.ty.to_ty(),
                out_ty = e.ty.to_ty(),
            ),
            ExprKind::Var(_) => panic!("[ICE] Attempted to generate MLIR for a var-assignment"),
            _ => unreachable!(),
        }
    }

    fn to_var(&self) -> String {
        match &self.kind {
            ExprKind::Var(id) => id.to_var(),
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
        use TypeKind::*;
        match self.kind {
            I8 => "i8".to_owned(),
            I16 => "i16".to_owned(),
            I32 => "i32".to_owned(),
            I64 => "i64".to_owned(),
            F32 => "f32".to_owned(),
            F64 => "f64".to_owned(),
            Bool => "i1".to_owned(),
            Null => todo!(),
            String => todo!(),
            Struct(_) => todo!(),
            Array(_, _) => todo!(),
            Tuple(_) => todo!(),
            Option(_) => todo!(),
            Unknown => "<UNKNOWN>".to_string(),
            Error => "<ERROR>".to_string(),
        }
    }
}

impl Lit {
    fn to_lit(&self) -> String {
        use Lit::*;
        match self {
            I32(l) => l.to_string(),
            I64(l) => l.to_string(),
            F32(l) => l.to_string(),
            F64(l) => l.to_string(),
            Bool(l) => l.to_string(),
            _ => todo!(),
        }
    }
}
