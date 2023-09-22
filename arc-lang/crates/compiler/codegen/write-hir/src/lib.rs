#![allow(unused)]

use codegen::*;
use colors::*;
use std::io::Result;
use std::io::Write;

use hir::*;

use im_rc::Vector;

pub fn write(ctx: &mut Context<impl Write>, ss: &Vector<Stmt>) -> Result<()> {
    ctx.each(ss.iter(), write_stmt)
}

pub fn write_stmt(ctx: &mut Context<impl Write>, s: &Stmt) -> Result<()> {
    match &s.kind {
        SVal(p, e) => {
            if let PVar(x) = p.kind.as_ref() {
                ctx.keyword("var ")?;
                write_name(ctx, x)?;
            } else {
                ctx.keyword("val ")?;
                write_pattern(ctx, p)?;
            }
            ctx.lit(" = ")?;
            write_expr(ctx, e)?;
            ctx.lit(";")?;
            ctx.newline()?;
        }
        SExpr(e) => {
            write_expr(ctx, e)?;
            ctx.lit(";")?;
            ctx.newline()?;
        }
        SDef(m, x, gs, ps, t, b) => {
            write_meta(ctx, &m)?;
            ctx.keyword("def ")?;
            ctx.colored(x, DEF_COLOR)?;
            write_generics(ctx, gs)?;
            write_patterns(ctx, ps)?;
            ctx.lit(":")?;
            write_type(ctx, t)?;
            ctx.lit(" ")?;
            write_block(ctx, b)?;
            ctx.newline()?;
        }
        SEnum(m, x, gs, vs) => {
            write_meta(ctx, &m)?;
            ctx.keyword("enum ")?;
            ctx.colored(x, TYPE_COLOR)?;
            write_generics(ctx, gs)?;
            ctx.lit(" ")?;
            ctx.brace(|ctx| {
                ctx.indented_comma_seq(vs, |ctx, (x, t)| {
                    write_name(ctx, x)?;
                    ctx.lit(":")?;
                    write_type(ctx, t)
                })
            })?;
            ctx.newline()?;
        }
        SBif(m, x, gs, ts, t) => {
            if ctx.opt.prelude {
                write_meta(ctx, &m)?;
                ctx.keyword("def ")?;
                ctx.colored(x, DEF_COLOR)?;
                write_generics(ctx, gs)?;
                ctx.paren(|ctx| ctx.seq(ts, write_type))?;
                ctx.lit(":")?;
                ctx.lit(" ")?;
                write_type(ctx, t)?;
                ctx.lit(";")?;
                ctx.newline()?;
            }
        }
        SBit(m, x, gs) => {
            if ctx.opt.prelude {
                write_meta(ctx, &m)?;
                ctx.keyword("type ")?;
                ctx.colored(x, TYPE_COLOR)?;
                write_generics(ctx, gs)?;
                ctx.lit(";")?;
                ctx.newline()?;
            }
        }
        SRecDef(_m, _ds) => todo!(),
        SNoop => {
            ctx.lit(";")?;
            ctx.newline()?;
        }
    }
    Ok(())
}

pub fn write_patterns(ctx: &mut Context<impl Write>, ps: &Vector<Pattern>) -> Result<()> {
    ctx.paren(|ctx| ctx.seq(ps, write_pattern))
}

pub fn write_name(ctx: &mut Context<impl Write>, x: &Name) -> Result<()> {
    ctx.lit(x)
}

pub fn write_generics(ctx: &mut Context<impl Write>, gs: &Vector<Generic>) -> Result<()> {
    if !gs.is_empty() {
        ctx.brack(|ctx| ctx.seq(gs, write_name))?;
    }
    Ok(())
}

pub fn write_types(ctx: &mut Context<impl Write>, ts: &Vector<Type>) -> Result<()> {
    if !ts.is_empty() {
        ctx.brack(|ctx| ctx.seq(ts, write_type))?;
    }
    Ok(())
}

pub fn write_qualified_types(ctx: &mut Context<impl Write>, ts: &Vector<Type>) -> Result<()> {
    if !ts.is_empty() {
        ctx.lit("::")?;
        ctx.brack(|ctx| ctx.seq(ts, write_type))?;
    }
    Ok(())
}

pub fn write_meta(ctx: &mut Context<impl Write>, d: &Meta) -> Result<()> {
    if !d.is_empty() {
        ctx.lit("@")?;
        ctx.brace(|ctx| {
            ctx.seq(d, |ctx, (k, v)| {
                write_name(ctx, k)?;
                ctx.then(v, |ctx, v| {
                    ctx.lit(":")?;
                    write_const(ctx, v)
                })
            })
        })?;
        ctx.newline()?;
    }
    Ok(())
}

pub fn write_pattern(ctx: &mut Context<impl Write>, p: &Pattern) -> Result<()> {
    fn write_pattern(ctx: &mut Context<impl Write>, p: &Pattern) -> Result<()> {
        match p.kind.as_ref() {
            PIgnore => {
                ctx.lit("_")?;
            }
            POr(p0, p1) => {
                write_pattern(ctx, p0)?;
                ctx.keyword(" or ")?;
                write_pattern(ctx, p1)?;
            }
            PNoop(p) => {
                write_pattern(ctx, p)?;
            }
            PRecord(xps) => {
                ctx.brace(|ctx| {
                    for (x, p) in xps {
                        write_name(ctx, x)?;
                        ctx.lit(":")?;
                        write_pattern(ctx, p)?;
                    }
                    Ok(())
                })?;
            }
            PArray(ps) => {
                ctx.brack(|ctx| {
                    for p in ps {
                        write_pattern(ctx, p)?;
                    }
                    Ok(())
                })?;
            }
            PConst(c) => {
                write_const(ctx, c)?;
            }
            PVariant(xs, ts, x, p) => {
                write_name(ctx, x)?;
                write_qualified_types(ctx, ts)?;
                ctx.lit(".")?;
                write_name(ctx, x)?;
                write_pattern(ctx, p)?;
                ctx.paren(|ctx| write_pattern(ctx, p))?;
            }
            PVar(x) => {
                ctx.colored(&x, VAR_COLOR)?;
                ctx.lit(":")?;
                write_type(ctx, &p.t)?;
            }
            PVal(x) => {
                ctx.colored(&x, VAL_COLOR)?;
                ctx.lit(":")?;
                write_type(ctx, &p.t)?;
            }
            PTuple(ps) => {
                ctx.paren(|ctx| ctx.seq_trailing(ps, write_pattern))?;
            }
            PError => {
                ctx.lit("<error>")?;
            }
            PRecordConcat(p0, p1) => {
                write_pattern(ctx, p0)?;
                ctx.lit(" & ")?;
                write_pattern(ctx, p1)?;
            }
            PArrayConcat(p0, p1) => {
                write_pattern(ctx, p0)?;
                ctx.lit(" ++ ")?;
                write_pattern(ctx, p1)?;
            }
        };
        Ok(())
    }
    if ctx.opt.types {
        ctx.paren(|ctx| {
            write_pattern(ctx, p)?;
            ctx.lit(":")?;
            write_type(ctx, &p.t)
        })
    } else {
        write_pattern(ctx, p)
    }
}

pub fn write_const(ctx: &mut Context<impl Write>, c: &Const) -> Result<()> {
    match c {
        CInt(c) => ctx.colored(&format!("{c}"), NUMERIC_COLOR),
        CFloat(c) => ctx.colored(&format!("{c}"), NUMERIC_COLOR),
        CString(c) => {
            ctx.colored("\"", STRING_COLOR)?;
            ctx.colored(c, STRING_COLOR)?;
            ctx.colored("\"", STRING_COLOR)
        }
        CBool(c) => ctx.bold_colored(&format!("{c}"), BUILTIN_COLOR),
        CChar(c) => ctx.colored(&format!("'{c}'"), STRING_COLOR),
        CUnit => ctx.lit("()"),
    }
}

pub fn write_expr(ctx: &mut Context<impl Write>, e: &Expr) -> Result<()> {
    fn write_expr(ctx: &mut Context<impl Write>, e: &Expr) -> Result<()> {
        match e.kind.as_ref() {
            EMut(e0, e1) => {
                write_expr(ctx, e0)?;
                ctx.lit(" = ")?;
                write_expr(ctx, e1)?;
            }
            ELoop(b) => {
                ctx.keyword("loop ")?;
                write_block(ctx, b)?;
            }
            ERecord(xes) => {
                ctx.brace(|ctx| {
                    ctx.seq(xes, |ctx, (x, e)| {
                        write_name(ctx, x)?;
                        ctx.lit(":")?;
                        write_expr(ctx, e)
                    })
                })?;
            }
            ERecordAccess(e, x) => {
                write_expr(ctx, e)?;
                ctx.lit(".")?;
                write_name(ctx, x)?;
            }
            ERecordConcat(e0, e1) => {
                write_expr(ctx, e0)?;
                ctx.lit(" ++ ")?;
                write_expr(ctx, e1)?;
            }
            EFunReturn(e) => {
                ctx.keyword("return")?;
                ctx.lit(" ")?;
                write_expr(ctx, e)?;
            }
            ELoopBreak(e) => {
                ctx.keyword("break")?;
            }
            ELoopContinue => {
                ctx.keyword("continue")?;
            }
            EMatch(e, arms) => {
                ctx.keyword("match ")?;
                write_expr(ctx, e)?;
                ctx.lit(" ")?;
                ctx.brace(|ctx| {
                    ctx.indented_seq(arms, |ctx, (p, b)| {
                        write_pattern(ctx, p)?;
                        ctx.lit(" => ")?;
                        ctx.keyword("do ")?;
                        write_block(ctx, b)?;
                        ctx.lit(",")
                    })
                })?;
            }
            EFunCall(e, es) => {
                write_expr(ctx, e)?;
                ctx.paren(|ctx| ctx.seq(es, write_expr))?;
            }
            EVariant(x0, ts, x1, e) => {
                write_name(ctx, x0)?;
                write_qualified_types(ctx, ts)?;
                ctx.lit("::")?;
                write_name(ctx, x1)?;
                ctx.paren(|ctx| write_expr(ctx, e))?;
            }
            EVariantAccess(x0, ts, x1, e) => {
                write_expr(ctx, e)?;
                ctx.lit(" as ")?;
                write_name(ctx, x0)?;
                write_qualified_types(ctx, ts)?;
                ctx.lit("::")?;
                write_name(ctx, x1)?;
            }
            EVariantCheck(x0, ts, x, e) => {
                write_expr(ctx, e)?;
                ctx.lit(" is ")?;
                write_name(ctx, x0)?;
                write_qualified_types(ctx, ts)?;
                ctx.lit("::")?;
                write_name(ctx, x)?;
            }
            EFun(ps, t, b) => {
                ctx.keyword("fun")?;
                ctx.paren(|ctx| ctx.seq(ps, write_pattern))?;
                ctx.lit(":")?;
                write_type(ctx, t)?;
                ctx.lit(" ")?;
                write_block(ctx, b)?;
            }
            EFunCallDirect(x, ts, es) => {
                ctx.colored(x, DEF_COLOR)?;
                write_qualified_types(ctx, ts)?;
                ctx.paren(|ctx| ctx.seq(es, write_expr))?;
            }
            EConst(c) => {
                write_const(ctx, c)?;
            }
            EIfElse(e, b0, b1) => {
                ctx.keyword("if ")?;
                write_expr(ctx, e)?;
                ctx.lit(" ")?;
                write_block(ctx, b0)?;
                ctx.keyword(" else ")?;
                write_block(ctx, b1)?;
            }
            ENoop(e) => {
                write_expr(ctx, e)?;
            }
            EDo(b) => {
                ctx.keyword("do ")?;
                write_block(ctx, b)?;
            }
            EArray(es) => {
                ctx.brack(|ctx| ctx.seq(es, write_expr))?;
            }
            EArrayConcat(e0, e1) => {
                write_expr(ctx, e0)?;
                ctx.lit(" ++ ")?;
                write_expr(ctx, e1)?;
            }
            EArrayAccess(e1, e2) => {
                write_expr(ctx, e1)?;
                ctx.brack(|ctx| write_expr(ctx, e2))?;
            }
            EVal(x) => {
                ctx.colored(&x, VAL_COLOR)?;
            }
            EVar(x) => {
                ctx.colored(&x, VAR_COLOR)?;
            }
            EDef(x, ts) => {
                ctx.colored(&x, DEF_COLOR)?;
                write_qualified_types(ctx, ts)?;
            }
            ETuple(es) => {
                ctx.paren(|ctx| ctx.seq_trailing(es, write_expr))?;
            }
            ETupleAccess(e, i) => {
                write_expr(ctx, e)?;
                ctx.lit(".")?;
                ctx.fmt(format_args!("{i}"))?;
            }
            EFor(p, e, b) => {
                ctx.keyword("for ")?;
                write_pattern(ctx, p)?;
                ctx.keyword(" in ")?;
                write_expr(ctx, e)?;
                ctx.lit(" ")?;
                write_block(ctx, b)?;
            }
            EWhile(e, b) => {
                ctx.keyword("while ")?;
                write_expr(ctx, e)?;
                ctx.lit(" ")?;
                write_block(ctx, b)?;
            }
            EError => {
                ctx.lit("<error>")?;
            }
        }
        Ok(())
    }
    if ctx.opt.types {
        ctx.paren(|ctx| {
            write_expr(ctx, e)?;
            ctx.lit(":")?;
            write_type(ctx, &e.t)
        })
    } else {
        write_expr(ctx, e)
    }
}

pub fn write_block(ctx: &mut Context<impl Write>, b: &Block) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.indent();
        ctx.newline();
        for s in &b.ss {
            write_stmt(ctx, s)?;
        }
        write_expr(ctx, &b.e)?;
        ctx.dedent();
        ctx.newline()
    })
}

pub fn write_type(ctx: &mut Context<impl Write>, t: &Type) -> Result<()> {
    match t.kind.as_ref() {
        TFun(ts, t) => {
            ctx.keyword("fun")?;
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
            ctx.lit(":")?;
            write_type(ctx, t)?;
        }
        TRecord(t) => {
            ctx.brace(|ctx| write_type(ctx, t))?;
        }
        TRowEmpty => {
            ctx.lit("Empty")?;
        }
        TRowExtend((x, t), r) => {
            ctx.lit("Row")?;
            ctx.paren(|ctx| {
                write_name(ctx, x)?;
                ctx.colon()?;
                write_type(ctx, t)?;
                ctx.comma()?;
                write_type(ctx, r)
            })?;
        }
        TRecordConcat(t0, t1) => {
            write_type(ctx, t0)?;
            ctx.lit(" & ")?;
            write_type(ctx, t1)?;
        }
        TNominal(x, ts) => {
            ctx.colored(x, TYPE_COLOR)?;
            write_types(ctx, ts)?;
        }
        TTuple(ts, closed) => {
            if *closed {
                ctx.paren(|ctx| ctx.seq_trailing(ts, write_type))?;
            } else {
                ctx.paren(|ctx| {
                    ctx.seq(ts, write_type)?;
                    ctx.lit(", ..")
                })?;
            }
        }
        TArray(t, n) => {
            ctx.brack(|ctx| {
                write_type(ctx, t)?;
                ctx.lit(";")?;
                ctx.then_or(
                    n,
                    |ctx, n| ctx.fmt(format_args!("{}", n)),
                    |ctx| ctx.lit("_"),
                )
            })?;
        }
        TArrayConcat(t0, t1) => {
            write_type(ctx, t0)?;
            ctx.lit(" ++ ")?;
            write_type(ctx, t1)?;
        }
        TGeneric(x) => {
            write_name(ctx, x)?;
        }
        TVar(x) => {
            ctx.lit("'")?;
            write_name(ctx, x)?;
        }
        TAlias(_, _, t) => {
            write_type(ctx, t)?;
        }
        TError => {
            ctx.lit("<Error>")?;
        }
        TUnit => {
            ctx.lit("()")?;
        }
        TNever => {
            ctx.lit("!")?;
        }
    }
    Ok(())
}

pub fn print_type(t: &Type) -> Result<()> {
    codegen::Context::stderr().typed().writeln(t, write_type)?;
    Ok(())
}

pub fn print_pattern(t: &Pattern) -> Result<()> {
    codegen::Context::stderr()
        .typed()
        .writeln(t, write_pattern)?;
    Ok(())
}

pub fn print_expr(e: &Expr) -> Result<()> {
    codegen::Context::stderr().typed().writeln(e, write_expr)?;
    Ok(())
}

pub fn print_stmt(s: &Stmt) -> Result<()> {
    codegen::Context::stderr().typed().writeln(s, write_stmt)?;
    Ok(())
}

pub fn print_block(b: &Block) -> Result<()> {
    codegen::Context::stderr().typed().writeln(b, write_block)?;
    Ok(())
}

pub fn type_to_string(t: &Type) -> String {
    codegen::Context::string()
        .write(t, write_type)
        .unwrap()
        .finish()
}
