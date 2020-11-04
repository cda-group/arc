use crate::codegen::printer::Printer;
use crate::prelude::*;
use std::cell::RefMut;

impl Script<'_> {
    /// MLIR code-generator.
    pub fn mlir(&self) -> String {
        let pr = Printer::from(&self.info);
        self.ast
            .fundefs
            .values()
            .map(|fundef| fundef.to_func(&pr))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Expr {
    pub fn mlir(&self, info: &Info) -> String {
        let pr = Printer::from(info);
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
            Let(id, e1, e2) => format!(
                "{s}{var} = {op}{next}",
                var = id.to_var(),
                op = e1.to_op(pr),
                next = e2.to_region(terminator, pr),
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

    #[rustfmt::skip]
    fn to_op(&self, pr: &Printer) -> String {
        match &self.kind {
            Lit(kind) => {
                let t = self.tv.to_ty(pr);
                let l = kind.to_lit();
                format!(r#""arc.constant"() {{ value = {l} : {t} }}: () -> {t}"#, l=l, t=t)
            }
            BinOp(l, op, r) => {
                let ty = pr.lookup(l.tv);
                let t = l.tv.to_ty(pr);
                let rt = self.tv.to_ty(pr);
                let l = l.to_var();
                let r = r.to_var();
                match (op, ty.kind) {
                    // Add
                    (Add, Scalar(I8 )) => format!(r#""arc.addi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    (Add, Scalar(I16)) => format!(r#""arc.addi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    (Add, Scalar(I32)) => format!(r#""arc.addi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    (Add, Scalar(I64)) => format!(r#""arc.addi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    (Add, Scalar(F32)) => format!(r#""std.addf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    (Add, Scalar(F64)) => format!(r#""std.addf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    // Sub
                    (Sub, Scalar(I8 )) => format!(r#""arc.subi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Sub, Scalar(I16)) => format!(r#""arc.subi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Sub, Scalar(I32)) => format!(r#""arc.subi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Sub, Scalar(I64)) => format!(r#""arc.subi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Sub, Scalar(F32)) => format!(r#""std.subf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Sub, Scalar(F64)) => format!(r#""std.subf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Mul
                    (Mul, Scalar(I8 )) => format!(r#""arc.muli"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Mul, Scalar(I16)) => format!(r#""arc.muli"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Mul, Scalar(I32)) => format!(r#""arc.muli"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Mul, Scalar(I64)) => format!(r#""arc.muli"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Mul, Scalar(F32)) => format!(r#""std.mulf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Mul, Scalar(F64)) => format!(r#""std.mulf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Div
                    (Div, Scalar(I8 )) => format!(r#""arc.divi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Div, Scalar(I16)) => format!(r#""arc.divi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Div, Scalar(I32)) => format!(r#""arc.divi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Div, Scalar(I64)) => format!(r#""arc.divi"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Div, Scalar(F32)) => format!(r#""std.divf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Div, Scalar(F64)) => format!(r#""std.divf"({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Lt 
                    (Lt, Scalar(I8 ))  => format!(r#""arc.cmpi "lt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Lt, Scalar(I16))  => format!(r#""arc.cmpi "lt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Lt, Scalar(I32))  => format!(r#""arc.cmpi "lt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Lt, Scalar(I64))  => format!(r#""arc.cmpi "lt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Lt, Scalar(F32))  => format!(r#""std.cmpf "olt" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Lt, Scalar(F64))  => format!(r#""std.cmpf "olt" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Leq 
                    (Leq, Scalar(I8 )) => format!(r#""arc.cmpi "le"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Leq, Scalar(I16)) => format!(r#""arc.cmpi "le"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Leq, Scalar(I32)) => format!(r#""arc.cmpi "le"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Leq, Scalar(I64)) => format!(r#""arc.cmpi "le"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Leq, Scalar(F32)) => format!(r#""std.cmpf "ole" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Leq, Scalar(F64)) => format!(r#""std.cmpf "ole" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Gt 
                    (Gt, Scalar(I8 ))  => format!(r#""arc.cmpi "gt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Gt, Scalar(I16))  => format!(r#""arc.cmpi "gt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Gt, Scalar(I32))  => format!(r#""arc.cmpi "gt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Gt, Scalar(I64))  => format!(r#""arc.cmpi "gt"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Gt, Scalar(F32))  => format!(r#""std.cmpf "ogt" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Gt, Scalar(F64))  => format!(r#""std.cmpf "ogt" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Geq
                    (Geq, Scalar(I8 )) => format!(r#""arc.cmpi "ge"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Geq, Scalar(I16)) => format!(r#""arc.cmpi "ge"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Geq, Scalar(I32)) => format!(r#""arc.cmpi "ge"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Geq, Scalar(I64)) => format!(r#""arc.cmpi "ge"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Geq, Scalar(F32)) => format!(r#""std.cmpf "oge" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Geq, Scalar(F64)) => format!(r#""std.cmpf "oge" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Equ
                    (Equ, Scalar(I8 )) => format!(r#""arc.cmpi "eq"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Equ, Scalar(I16)) => format!(r#""arc.cmpi "eq"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Equ, Scalar(I32)) => format!(r#""arc.cmpi "eq"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Equ, Scalar(I64)) => format!(r#""arc.cmpi "eq"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Equ, Scalar(F32)) => format!(r#""std.cmpf "oeq" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Equ, Scalar(F64)) => format!(r#""std.cmpf "oeq" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // Neq
                    (Neq, Scalar(I8 )) => format!(r#""arc.cmpi "ne"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Neq, Scalar(I16)) => format!(r#""arc.cmpi "ne"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Neq, Scalar(I32)) => format!(r#""arc.cmpi "ne"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Neq, Scalar(I64)) => format!(r#""arc.cmpi "ne"  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Neq, Scalar(F32)) => format!(r#""std.cmpf "one" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    (Neq, Scalar(F64)) => format!(r#""std.cmpf "one" ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt), 
                    // And
                    (And, _)           => format!(r#""arc.and ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    // Or
                    (Or,  _)           => format!(r#""arc.or  ({l},{r}) : ({t},{t}) -> {rt}"#, l=l, r=r, t=t, rt=rt),
                    x => todo!("{:?}", x),
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
            UnOp(kind, e) => {
                match (kind, &e.kind) {
                    (Neg, Lit(lit)) => {
                        let t = self.tv.to_ty(pr);
                        let l = lit.to_lit();
                        match lit {
                            LitI8(_)  => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            LitI16(_) => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            LitI32(_) => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            LitI64(_) => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            LitF32(_) => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            LitF64(_) => format!(r#""arc.constant"() {{ value = -{l} : {t} }}: () -> {t}"#, l=l, t=t),
                            _ => unreachable!()
                        }
                    }
                    (Call(args), Var(id)) => {
                        let ty = e.tv.to_ty(pr);
                        let id = id.to_var();
                        let args = args.iter().map(|arg| arg.to_var()).collect::<Vec<_>>().join(",");
                        format!(r#"call @{id}({args}) {ty}"#, id=id, args=args, ty=ty)
                    },
                    x => todo!("{:?}", x)
                }
            }
            ConsArray(..) => todo!(),
            ConsStruct(..) => todo!(),
            ConsVariant(_, _) => todo!(),
            ConsTuple(..) => todo!(),
            Closure(..) => todo!(),
            Let(..) => panic!("[ICE] Attempted to generate MLIR SSA of Let"),
            Match(..) => panic!("[ICE] Attempted to generate MLIR SSA of Match"),
            Var(_) => panic!("[ICE] Attempted to generate MLIR SSA of Var"),
            Loop(_, _) => todo!(),
            For(..) => todo!(),
            ExprErr => "<EXPR-ERROR>".to_owned(),
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
        let mut typer = pr.info.typer.borrow_mut();
        self.to_ty_rec(&mut typer)
    }
    #[rustfmt::skip]
    fn to_ty_rec(self, typer: &mut RefMut<Typer>) -> String {
        match typer.lookup(self).kind {
            Nominal(_) => todo!(),
            Scalar(kind) => match kind {
                I8   => "i8".to_owned(),
                I16  => "i16".to_owned(),
                I32  => "i32".to_owned(),
                I64  => "i64".to_owned(),
                F32  => "f32".to_owned(),
                F64  => "f64".to_owned(),
                Bool => "i1".to_owned(),
                Null => todo!(),
                Str  => todo!(),
                Unit => "()".to_owned(),
            },
            Struct(_)      => todo!(),
            Enum(_)        => todo!(),
            Array(_, _)    => todo!(),
            Stream(_)      => todo!(),
            Map(_, _)      => todo!(),
            Set(_)         => todo!(),
            Vector(_)      => todo!(),
            Tuple(_)       => todo!(),
            Optional(_)    => todo!(),
            Fun(ptvs, rtv) => {
                let ptys = ptvs.iter().map(|tv| tv.to_ty_rec(typer)).collect::<Vec<_>>().join(",");
                let rty = rtv.to_ty_rec(typer);
                format!("({ptys}) -> {rty}", ptys=ptys, rty=rty)
            },
            Task(_)        => todo!(),
            Unknown        => "<UNKNOWN>".to_string(),
            TypeErr        => "<TYPE-ERROR>".to_string(),
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
            LitErr => "<LIT-ERROR>".to_string(),
        }
    }
}

impl FunDef {
    fn to_func(&self, pr: &Printer) -> String {
        let tv = pr.info.table.get_decl(&self.id).tv;
        let ty = pr.lookup(tv);
        if let Fun(_, ret_tv) = ty.kind {
            format!(
                "func @{id}({params}) -> ({ret_ty}) {{{body}}}",
                id = self.id.to_var(),
                params = self
                    .params
                    .iter()
                    .map(|id| pr.info.table.get_decl(id).tv.to_ty(pr))
                    .collect::<Vec<_>>()
                    .join(","),
                ret_ty = ret_tv.to_ty(&pr),
                body = self.body.to_region("std.return", &pr.tab()),
            )
        } else {
            "<FUNC-ERROR>".to_string()
        }
    }
}
