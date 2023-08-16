#![allow(unused)]
use codegen::Context;
use std::io::Result;
use std::io::Write;

use mlir::*;

use im_rc::Vector;

pub fn write(ctx: &mut Context<impl Write>, is: &Vector<Item>) -> Result<()> {
    ctx.each_newline(is.iter(), write_item)
}

pub fn write_module(ctx: &mut Context<impl Write>, is: &Vector<Item>) -> Result<()> {
    ctx.keyword("module")?;
    ctx.space()?;
    write_symbol(ctx, "@program")?;
    ctx.space()?;
    ctx.brace(|ctx| {
        ctx.newline()?;
        ctx.indent();
        for line in ast_prelude::MLIR_PRELUDE.lines() {
            ctx.newline()?;
            ctx.lit(line)?;
        }
        ctx.newline()?;
        ctx.each_newline(is.iter(), write_item)?;
        ctx.dedent();
        ctx.newline()
    })
}

pub fn write_item(ctx: &mut Context<impl Write>, i: &Item) -> Result<()> {
    match &i.kind {
        IDef(x, xvs, t, b) => {
            ctx.keyword("func.func")?;
            ctx.space()?;
            write_symbol(ctx, x)?;
            write_params(ctx, xvs)?;
            ctx.colon()?;
            write_type(ctx, t)?;
            ctx.lit(" ")?;
            write_block(ctx, b)?;
        }
        IExternDef(x, xvs, t) => {
            ctx.keyword("func.func")?;
            ctx.space()?;
            ctx.keyword("private")?;
            ctx.space()?;
            write_symbol(ctx, x)?;
            write_params(ctx, xvs)?;
            ctx.colon()?;
            write_type(ctx, t)?;
        }
        // IDataflow(d) => todo!(),
        IError => unreachable!(),
    }
    Ok(())
}

pub fn write_params(ctx: &mut Context<impl Write>, ps: &Vector<Val>) -> Result<()> {
    ctx.paren(|ctx| {
        ctx.seq(ps, |ctx, v| {
            write_val(ctx, v)?;
            ctx.colon()?;
            write_type(ctx, &v.t)
        })
    })?;
    Ok(())
}

pub fn write_stmt(ctx: &mut Context<impl Write>, s: &Stmt) -> Result<()> {
    if !s.vs.is_empty() {
        ctx.seq(&s.vs, write_val)?;
        ctx.lit(" = ")?;
    }
    match &s.kind {
        SRecord(xvs) => {
            ctx.keyword("arc.make_struct")?;
            ctx.paren(|ctx| {
                ctx.seq(xvs, |ctx, (_, v)| write_val(ctx, v))?;
                ctx.colon()?;
                ctx.seq(xvs, |ctx, (_, v)| write_typeof(ctx, v))?;
                Ok(())
            })?;
            ctx.colon()?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SRecordAccess(v, x) => {
            ctx.quote(|ctx| ctx.lit("arc.struct_access"))?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.brace(|ctx| {
                ctx.lit("field")?;
                ctx.lit(" = ")?;
                ctx.quote(|ctx| ctx.lit(x))
            })?;
            ctx.colon()?;
            ctx.paren(|ctx| write_type(ctx, &v.t))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SWhileBreak(vs) => {
            ctx.quote(|ctx| ctx.lit("arc.loop.break"))?;
            ctx.paren(|ctx| ctx.seq(vs, write_val))?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SWhileContinue(vs) => {
            ctx.quote(|ctx| ctx.lit("arc.loop.continue"))?;
            ctx.paren(|ctx| ctx.seq(vs, write_val))?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SWhileYield(vs) => {
            ctx.quote(|ctx| ctx.lit("scf.yield"))?;
            ctx.paren(|ctx| ctx.seq(vs, write_val))?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SFunCallDirect(x, vs) => {
            ctx.lit("call ")?;
            write_symbol(ctx, x)?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SFunCallIndirect(v, vs) => {
            ctx.lit("call_indirect ")?;
            write_val(ctx, v)?;
            ctx.paren(|ctx| ctx.seq(vs, write_val))?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SVariant(x, v) => {
            ctx.lit("arc.make_enum")?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.lit(" as ")?;
            ctx.quote(|ctx| ctx.lit(x))?;
            ctx.colon()?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SVariantCheck(x, v) => {
            ctx.lit("arc.check_enum")?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.lit(" in ")?;
            ctx.quote(|ctx| ctx.lit(x))?;
            ctx.colon()?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SVariantAccess(x, v) => {
            ctx.lit("arc.enum_access")?;
            ctx.quote(|ctx| ctx.lit(x))?;
            ctx.lit(" in ")?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.colon()?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SFun(x) => {
            ctx.lit("constant ")?;
            write_symbol(ctx, x)?;
            ctx.colon()?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SConst(c) => match c {
            CInt(c) => {
                ctx.lit("arc.constant ")?;
                ctx.fmt(format_args!("{c}"))?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
            CFloat(c) => {
                ctx.lit("arith.constant ")?;
                ctx.fmt(format_args!("{c}"))?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
            CString(c) => {
                ctx.lit("arc.adt_constant ")?;
                ctx.fmt(format_args!(r#"String::new("{c}")"#))?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
            CBool(c) => {
                ctx.lit("arith.constant ")?;
                ctx.fmt(format_args!("{c}"))?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
            CChar(c) => {
                ctx.lit("arc.adt_constant ")?;
                ctx.fmt(format_args!("'{c}'"))?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
            CUnit => {
                ctx.lit("arc.make_struct()")?;
                ctx.colon()?;
                ctx.seq(&s.vs, write_typeof)?;
            }
        },
        SIfElse(v, b0, b1) => {
            ctx.lit("arc.if")?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.paren(|ctx| {
                write_block(ctx, b0)?;
                ctx.lit(",")?;
                write_block(ctx, b1)
            })?;
            ctx.colon()?;
            ctx.paren(|ctx| write_type(ctx, &v.t))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SFunReturn(v) => {
            ctx.lit("return ")?;
            write_val(ctx, v)?;
            ctx.colon()?;
            write_type(ctx, &v.t)?;
        }
        SBlockResult(v) => {
            ctx.quote(|ctx| ctx.lit("arc.block.result"))?;
            ctx.paren(|ctx| write_val(ctx, v))?;
            ctx.colon()?;
            ctx.paren(|ctx| write_type(ctx, &v.t))?;
            ctx.lit(" -> ")?;
            ctx.seq(&s.vs, write_typeof)?;
        }
        SWhile(vs0, vs1, b0, b1) => {
            ctx.lit("scf.while")?;
            ctx.paren(|ctx| {
                ctx.seq(vs0.into_iter().zip(vs1), |ctx, (v0, v1)| {
                    write_val(ctx, v0)?;
                    ctx.lit(" = ")?;
                    write_val(ctx, v1)
                })
            })?;
            ctx.colon()?;
            ctx.paren(|ctx| ctx.seq(vs0, write_typeof))?;
            ctx.lit(" -> ")?;
            ctx.paren(|ctx| ctx.seq(vs1, write_typeof))?;
            ctx.brace(|ctx| {
                write_block(ctx, b0)?;
                ctx.lit(",")?;
                write_block(ctx, b1)
            })?;
            ctx.lit(" do ")?;
            ctx.brace(|ctx| {
                ctx.lit("bb0^")?;
                ctx.paren(|ctx| ctx.seq(vs0, write_val))?;
                ctx.indent();
                for s in &b1.ss {
                    ctx.newline()?;
                    write_stmt(ctx, s)?;
                }
                ctx.dedent();
                ctx.newline()
            })?;
        }
        SError => unreachable!(),
    }
    Ok(())
}

pub fn write_block(ctx: &mut Context<impl Write>, b: &Block) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.indent();
        for s in &b.ss {
            ctx.newline()?;
            write_stmt(ctx, s)?;
        }
        ctx.dedent();
        ctx.newline()
    })
}

pub fn write_type(ctx: &mut Context<impl Write>, t: &Type) -> Result<()> {
    match t.kind.as_ref() {
        TFun(ts, t) => {
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
            ctx.lit(" -> ")?;
            write_type(ctx, t)?;
        }
        TRecord(xts) => {
            ctx.lit("!arc.struct")?;
            ctx.angle(|ctx| {
                ctx.seq(xts, |ctx, (x, t)| {
                    ctx.lit(x)?;
                    ctx.lit(": ")?;
                    write_type(ctx, t)
                })
            })?;
        }
        TNative(x, ts) => {
            ctx.lit(x)?;
            if !ts.is_empty() {
                ctx.angle(|ctx| ctx.seq(ts, write_type))?;
            }
        }
        TEnum(xts) => {
            ctx.lit("!arc.enum")?;
            ctx.angle(|ctx| {
                ctx.seq(xts, |ctx, (x, t)| {
                    ctx.lit(x)?;
                    ctx.lit(": ")?;
                    write_type(ctx, t)
                })
            })?;
        }
        TAdt(x, ts) => {
            if ts.is_empty() {
                ctx.lit("!arc.adt")?;
                ctx.angle(|ctx| ctx.quote(|ctx| ctx.lit(x)))?;
            } else {
                ctx.lit("!arc.generic_adt")?;
                ctx.angle(|ctx| {
                    ctx.quote(|ctx| ctx.lit(x))?;
                    ctx.lit(", ")?;
                    ctx.seq(ts, write_type)
                })?;
            }
        }
        TError => unreachable!(),
    }
    Ok(())
}

pub fn write_val(ctx: &mut Context<impl Write>, v: &Val) -> Result<()> {
    match &v.kind {
        VName(x) => {
            ctx.lit("%")?;
            ctx.lit(&x)
        }
        VError => unreachable!(),
    }
}

pub fn write_typeof(ctx: &mut Context<impl Write>, v: &Val) -> Result<()> {
    write_type(ctx, &v.t)
}

pub fn write_symbol(ctx: &mut Context<impl Write>, x: &str) -> Result<()> {
    ctx.def("@")?;
    ctx.def(&x)
}
