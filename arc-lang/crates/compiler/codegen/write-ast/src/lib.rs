#![allow(unused)]

use codegen::Context;
use std::io::Result;
use std::io::Write;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;

use ast::*;

use im_rc::Vector;

pub fn write(ctx: &mut Context<impl Write>, ss: &Vector<Stmt>) -> Result<()> {
    ctx.each(ss, write_stmt)
}

pub fn write_stmt(ctx: &mut Context<impl Write>, s: &Stmt) -> Result<()> {
    match &s.kind {
        SDef(m, x, gs, ps, t, bs, b) => {
            write_meta(ctx, m)?;
            ctx.keyword("def")?;
            ctx.space()?;
            ctx.def(x)?;
            write_generics(ctx, gs)?;
            write_patterns(ctx, ps)?;
            ctx.then(t, |ctx, t| {
                ctx.colon()?;
                write_type(ctx, t)
            })?;
            write_bounds(ctx, bs)?;
            ctx.space()?;
            match b {
                BExpr(e) => {
                    ctx.lit("=")?;
                    ctx.space()?;
                    write_expr(ctx, e)?;
                    ctx.lit(";")?;
                }
                BBlock(b) => {
                    write_block(ctx, b)?;
                }
            }
        }
        SEnum(m, x, gs, bs, xts) => {
            write_meta(ctx, m)?;
            ctx.keyword("enum")?;
            ctx.space()?;
            ctx.lit(x)?;
            ctx.space()?;
            write_generics(ctx, gs)?;
            write_bounds(ctx, bs)?;
            ctx.brace(|ctx| ctx.indented_comma_seq(xts, write_variant))?;
        }
        SType(m, x, gs, t) => {
            write_meta(ctx, m)?;
            ctx.keyword("type")?;
            ctx.space()?;
            ctx.def(x)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_generics(ctx, gs)?;
            write_type(ctx, t)?;
            ctx.lit(";")?;
        }
        SNoop => {
            ctx.lit(";")?;
        }
        SVal(p, e) => {
            ctx.keyword("val")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.lit(";")?;
        }
        SVar(p, e) => {
            ctx.keyword("var")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.lit(";")?;
        }
        SExpr(e) => {
            write_expr(ctx, e)?;
            ctx.lit(";")?;
        }
        SBuiltinDef(m, x, gs, ts, t, bs) => {
            // write_meta(ctx, m)?;
            ctx.keyword("def")?;
            ctx.space()?;
            ctx.def(x)?;
            write_generics(ctx, gs)?;
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
            ctx.then(t, |ctx, t| {
                ctx.colon()?;
                write_type(ctx, t)
            })?;
            write_bounds(ctx, bs)?;
            ctx.lit(";")?;
        }
        SBuiltinType(m, x, gs, bs) => {
            // write_meta(ctx, m)?;
            ctx.keyword("type")?;
            ctx.space()?;
            ctx.def(x)?;
            write_generics(ctx, gs)?;
            write_bounds(ctx, bs)?;
            ctx.lit(";")?;
        }
        SBuiltinClass(m, x, gs, bs) => {
            write_meta(ctx, m)?;
            ctx.keyword("trait")?;
            ctx.space()?;
            ctx.lit(x)?;
            write_generics(ctx, gs)?;
            write_bounds(ctx, bs)?;
            ctx.lit(";")?;
        }
        SBuiltinInstance(m, x, gs, bs, t) => {
            write_meta(ctx, m)?;
            ctx.keyword("impl")?;
            ctx.space()?;
            ctx.lit(x)?;
            write_generics(ctx, gs)?;
            write_bounds(ctx, bs)?;
            ctx.space()?;
            ctx.keyword("for")?;
            ctx.space()?;
            write_type(ctx, t)?;
            ctx.space()?;
            ctx.lit(";")?;
        }
        SInject(lang, code) => {
            ctx.comment("---")?;
            ctx.lit(lang)?;
            ctx.lit(code)?;
        }
    }
    ctx.newline()?;
    Ok(())
}

pub fn write_variant(ctx: &mut Context<impl Write>, v: &Variant) -> Result<()> {
    match v {
        VUnit(x) => {
            ctx.lit(x)?;
        }
        VRecord(x, xts) => {
            ctx.lit(x)?;
            ctx.space()?;
            write_type_record(ctx, xts)?;
        }
        VTuple(x, ts) => {
            ctx.lit(x)?;
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
        }
    }
    Ok(())
}

pub fn write_type_record(ctx: &mut Context<impl Write>, xts: &Vector<(Name, Type)>) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.seq(xts, |ctx, (x, t)| {
            ctx.lit(x)?;
            ctx.lit(":")?;
            ctx.space()?;
            write_type(ctx, t)
        })
    })
}

pub fn write_patterns(ctx: &mut Context<impl Write>, ps: &Vector<Pattern>) -> Result<()> {
    ctx.paren(|ctx| ctx.seq(ps, |ctx, p| write_pat(ctx, p)))?;
    Ok(())
}

pub fn write_bounds(ctx: &mut Context<impl Write>, bs: &Vector<Bound>) -> Result<()> {
    if !bs.is_empty() {
        ctx.keyword("where")?;
        ctx.space()?;
        ctx.lit(":")?;
        ctx.space()?;
        ctx.seq(bs, |ctx, (x, ts)| {
            ctx.lit(x)?;
            ctx.brack(|ctx| ctx.seq(ts, write_type))
        })?;
    }
    Ok(())
}

pub fn write_generics(ctx: &mut Context<impl Write>, gs: &Vector<Generic>) -> Result<()> {
    if !gs.is_empty() {
        ctx.brack(|ctx| ctx.seq(gs, |ctx, x| ctx.ty(x)))
    } else {
        Ok(())
    }
}

pub fn write_source(ctx: &mut Context<impl Write>, (p, e): &(Pattern, Expr)) -> Result<()> {
    write_pat(ctx, p)?;
    ctx.space()?;
    ctx.keyword("in")?;
    ctx.space()?;
    write_expr(ctx, e)
}

pub fn write_meta(ctx: &mut Context<impl Write>, m: &Meta) -> Result<()> {
    if !m.is_empty() {
        ctx.lit("@")?;
        ctx.brace(|ctx| ctx.seq(m, write_attr))?;
        ctx.newline()?;
    }
    Ok(())
}

pub fn write_attr(ctx: &mut Context<impl Write>, a: &Attr) -> Result<()> {
    ctx.lit(&a.x)?;
    ctx.then(&a.c, |ctx, c| {
        ctx.lit(":")?;
        write_const(ctx, c)
    })
}

pub fn write_pat(ctx: &mut Context<impl Write>, p: &Pattern) -> Result<()> {
    match p.kind.as_ref() {
        PParen(p) => {
            ctx.paren(|ctx| write_pat(ctx, p))?;
        }
        PIgnore => {
            ctx.lit("_")?;
        }
        POr(p0, p1) => {
            write_pat(ctx, p0)?;
            ctx.space()?;
            ctx.keyword("or")?;
            ctx.space()?;
            write_pat(ctx, p1)?;
        }
        PTypeAnnot(p, t) => {
            write_pat(ctx, p)?;
            ctx.lit(":")?;
            ctx.space()?;
            write_type(ctx, t)?;
        }
        PRecord(xps) => {
            write_pat_record(ctx, xps)?;
        }
        PRecordConcat(p0, p1) => {
            write_pat(ctx, p0)?;
            ctx.space()?;
            ctx.lit("&")?;
            ctx.space()?;
            write_pat(ctx, p1)?;
        }
        PTuple(ps) => {
            ctx.paren(|ctx| ctx.seq(ps, write_pat))?;
        }
        PArray(ps) => {
            ctx.brack(|ctx| ctx.seq(ps, write_pat))?;
        }
        PArrayConcat(p0, p1) => {
            write_pat(ctx, p0)?;
            ctx.space()?;
            ctx.lit("++")?;
            ctx.space()?;
            write_pat(ctx, p1)?;
        }
        PConst(c) => {
            write_const(ctx, c)?;
        }
        PName(x) => {
            ctx.lit(x)?;
        }
        PVariantTuple(x, ps) => {
            ctx.lit(x)?;
            ctx.paren(|ctx| ctx.seq(ps, write_pat))?;
        }
        PVariantRecord(x, xps) => {
            ctx.lit(x)?;
            write_pat_record(ctx, xps)?;
        }
        PError => {
            ctx.lit("<error>")?;
        }
    };
    Ok(())
}

pub fn write_pat_record(
    ctx: &mut Context<impl Write>,
    xps: &Vector<(Name, Option<Pattern>)>,
) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.seq(xps, |ctx, (x, p)| {
            ctx.lit(x)?;
            ctx.then(p, |ctx, p| {
                ctx.lit(":")?;
                ctx.space()?;
                write_pat(ctx, p)
            })
        })
    })
}

pub fn write_const(ctx: &mut Context<impl Write>, c: &Const) -> Result<()> {
    match c {
        CInt(i) => ctx.fmt(format_args!("{i}")),
        CFloat(f) => ctx.fmt(format_args!("{f}")),
        CString(s) => ctx.fmt(format_args!(r#""{s}""#)),
        CBool(b) => ctx.fmt(format_args!("{b}")),
        CUnit => ctx.lit("()"),
        CChar(c) => ctx.fmt(format_args!("'{c}'")),
    }
}

pub fn write_lit(ctx: &mut Context<impl Write>, c: &Lit) -> Result<()> {
    match c {
        LInt(i, s) => {
            ctx.fmt(format_args!("{i}"))?;
            ctx.then(s, |ctx, s| ctx.lit(s))
        }
        LFloat(f, s) => {
            ctx.fmt(format_args!("{f}"))?;
            ctx.then(s, |ctx, s| ctx.lit(s))
        }
        LString(s) => ctx.fmt(format_args!(r#""{s}""#)),
        LBool(b) => ctx.fmt(format_args!("{b}")),
        LUnit => ctx.lit("()"),
        LChar(c) => ctx.fmt(format_args!("'{c}'")),
    }
}

pub fn write_expr(ctx: &mut Context<impl Write>, e: &Expr) -> Result<()> {
    match e.kind.as_ref() {
        EParen(e) => {
            ctx.paren(|ctx| write_expr(ctx, e))?;
        }
        EQuery(p, e, qs) => {
            ctx.keyword("from")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.newline_seq(qs, write_query_stmt)?;
        }
        ERecordAccess(e, x) => {
            write_expr(ctx, e)?;
            ctx.lit(".")?;
            ctx.lit(x)?;
        }
        ERecordAccessMulti(e, xs) => {
            write_expr(ctx, e)?;
            ctx.lit(".")?;
            ctx.brace(|ctx| ctx.seq(xs, |ctx, x| ctx.lit(x)))?;
        }
        ERecordConcat(e0, e1) => {
            write_expr(ctx, e0)?;
            ctx.space()?;
            ctx.lit("&")?;
            ctx.space()?;
            write_expr(ctx, e1)?;
        }
        EFunCall(e, es) => {
            write_expr(ctx, e)?;
            ctx.paren(|ctx| ctx.seq(es, write_expr))?;
        }
        ETypeAnnot(e, t) => {
            write_expr(ctx, e)?;
            ctx.lit(":")?;
            ctx.space()?;
            write_type(ctx, t)?;
        }
        EIfElse(e, b0, b1) => {
            ctx.keyword("if")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            write_block(ctx, b0)?;
            ctx.then(b1, |ctx, b1| {
                ctx.space()?;
                ctx.keyword("else")?;
                ctx.space()?;
                write_block(ctx, b1)
            })?;
        }
        ELit(l) => {
            write_lit(ctx, l)?;
        }
        ELoop(b) => {
            ctx.keyword("loop")?;
            ctx.space()?;
            write_block(ctx, b)?;
        }
        ERecord(xes) => {
            write_expr_record(ctx, xes)?;
        }
        EVariantRecord(x, xes) => {
            ctx.lit(x)?;
            ctx.space()?;
            write_expr_record(ctx, xes)?;
        }
        EFunReturn(e) => {
            ctx.keyword("return")?;
            ctx.then(e, |ctx, e| {
                ctx.space()?;
                write_expr(ctx, e)
            })?;
        }
        ELoopBreak(e) => {
            ctx.keyword("break")?;
            ctx.then(e, |ctx, e| {
                ctx.space()?;
                write_expr(ctx, e)
            })?;
        }
        ELoopContinue => {
            ctx.keyword("continue")?;
        }
        EUnop(op, e) => {
            write_unop(ctx, op)?;
            write_expr(ctx, e)?;
        }
        EArray(es) => {
            ctx.brack(|ctx| ctx.seq(es, write_expr))?;
        }
        EArrayConcat(e0, e1) => {
            write_expr(ctx, e0)?;
            ctx.space()?;
            ctx.lit("++")?;
            ctx.space()?;
            write_expr(ctx, e1)?;
        }
        EArrayAccess(e, es) => {
            write_expr(ctx, e)?;
            ctx.brack(|ctx| write_expr(ctx, e))?;
        }
        EBinop(e0, op, e1) => {
            write_expr(ctx, e0)?;
            ctx.space()?;
            write_binop(ctx, op)?;
            ctx.space()?;
            write_expr(ctx, e1)?;
        }
        EMut(e0, op, e1) => {
            write_expr(ctx, e0)?;
            ctx.then(op, write_binop)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e1)?;
        }
        EDo(b) => {
            ctx.keyword("do")?;
            ctx.space()?;
            write_block(ctx, b)?;
        }
        EFor(p, e, b) => {
            ctx.keyword("for")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            write_block(ctx, b)?;
        }
        EFun(ps, t, b) => {
            ctx.keyword("fun")?;
            ctx.paren(|ctx| ctx.seq(ps, write_pat))?;
            ctx.then(t, |ctx, t| {
                ctx.lit(":")?;
                ctx.space()?;
                write_type(ctx, t)
            })?;
            ctx.space()?;
            match b {
                BExpr(e) => {
                    ctx.lit("=")?;
                    ctx.space()?;
                    write_expr(ctx, e)?;
                }
                BBlock(b) => {
                    write_block(ctx, b)?;
                }
            }
        }
        EMethodCall(e, x, ts, es) => {
            write_expr(ctx, e)?;
            ctx.lit(".")?;
            ctx.lit(x)?;
            write_qualified_types(ctx, ts)?;
            ctx.paren(|ctx| ctx.seq(es, write_expr))?;
        }
        EMatch(e, arms) => {
            ctx.keyword("match")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            ctx.brace(|ctx| write_arms(ctx, arms))?;
        }
        EName(x, ts) => {
            ctx.lit(x)?;
            write_qualified_types(ctx, ts)?;
        }
        ETupleAccess(e, i) => {
            write_expr(ctx, e)?;
            ctx.lit(".")?;
            ctx.fmt(format_args!("{i}"))?;
        }
        EThrow(e) => {
            ctx.keyword("throw")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        ETry(b0, arms, b1) => {
            ctx.keyword("try")?;
            ctx.space()?;
            write_block(ctx, b0)?;
            write_arms(ctx, arms)?;
            ctx.then(b1, |ctx, b1| {
                ctx.space()?;
                ctx.keyword("finally")?;
                ctx.space()?;
                write_block(ctx, b1)
            })?;
        }
        ETuple(es) => {
            ctx.paren(|ctx| ctx.seq(es, write_expr))?;
        }
        EWhile(e, b) => {
            ctx.keyword("while")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            write_block(ctx, b)?;
        }
        EError => {
            ctx.lit("<ERROR>")?;
        }
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

pub fn write_expr_record(ctx: &mut Context<impl Write>, xes: &Vector<ExprField>) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.seq(xes, |ctx, xe| match xe {
            FName(x, e) => {
                ctx.lit(x)?;
                ctx.then(e, |ctx, e| {
                    ctx.lit(":")?;
                    ctx.space()?;
                    write_expr(ctx, e)
                })
            }
            FExpr(e, x) => {
                write_expr(ctx, e)?;
                ctx.lit(".")?;
                ctx.lit(x)
            }
        })
    })
}

pub fn write_arms(ctx: &mut Context<impl Write>, arms: &Vector<Arm>) -> Result<()> {
    ctx.indented_comma_seq(arms, |ctx, (p, e)| {
        write_pat(ctx, &p)?;
        ctx.space()?;
        ctx.lit("=>")?;
        ctx.space()?;
        write_expr(ctx, &e)
    })
}

pub fn write_query_stmt(ctx: &mut Context<impl Write>, s: &QueryStmt) -> Result<()> {
    match s {
        QFrom(p, e) => {
            ctx.keyword("from")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QWhere(e) => {
            ctx.keyword("where")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QWith(p, e) => {
            ctx.keyword("with")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QJoinOn(p, e0, e1) => {
            ctx.keyword("join")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e0)?;
            ctx.space()?;
            ctx.keyword("on")?;
            ctx.space()?;
            write_expr(ctx, e1)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
        }
        QJoinOver(p, e, e0) => {
            ctx.keyword("join")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            ctx.keyword("over")?;
            ctx.space()?;
            write_expr(ctx, e0)?;
        }
        QJoinOverOn(p, e, e0, e1, e2, qs) => {
            ctx.keyword("join")?;
            ctx.space()?;
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.keyword("in")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            ctx.keyword("over")?;
            ctx.space()?;
            write_expr(ctx, e0)?;
            ctx.space()?;
            ctx.keyword("on")?;
            ctx.space()?;
            write_expr(ctx, e1)?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e2)?;
            ctx.space()?;
            ctx.brace(|ctx| ctx.newline_seq(qs, write_query_stmt))?;
        }
        QGroup(e, qs, x) => {
            ctx.keyword("group")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            ctx.brace(|ctx| ctx.newline_seq(qs, write_query_stmt))?;
            ctx.then(x, |ctx, x| {
                ctx.space()?;
                ctx.keyword("as")?;
                ctx.space()?;
                ctx.lit(x)
            })?;
        }
        QCompute(e0, e1, x) => {
            ctx.keyword("compute")?;
            ctx.space()?;
            write_expr(ctx, e0)?;
            ctx.then(e1, |ctx, e1| {
                ctx.space()?;
                ctx.keyword("of")?;
                ctx.space()?;
                write_expr(ctx, e1)
            })?;
            ctx.then(x, |ctx, x| {
                ctx.space()?;
                ctx.keyword("as")?;
                ctx.space()?;
                ctx.lit(x)
            })?;
        }
        QOver(e, qs, x) => {
            ctx.keyword("over")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            ctx.brace(|ctx| ctx.newline_seq(qs, write_query_stmt))?;
            ctx.then(x, |ctx, x| {
                ctx.space()?;
                ctx.keyword("as")?;
                ctx.space()?;
                ctx.lit(x)
            })?;
        }
        QRoll(e0, e1, x) => {
            ctx.keyword("roll")?;
            ctx.space()?;
            write_expr(ctx, e0)?;
            ctx.then(e1, |ctx, e1| {
                ctx.space()?;
                ctx.keyword("of")?;
                ctx.space()?;
                write_expr(ctx, e1)
            })?;
            ctx.then(x, |ctx, x| {
                ctx.space()?;
                ctx.keyword("as")?;
                ctx.space()?;
                ctx.lit(x)
            })?;
        }
        QSelect(e) => {
            ctx.keyword("select")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QUnion(e) => {
            ctx.keyword("union")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QInto(e) => {
            ctx.keyword("into")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QVal(p, e) => {
            write_pat(ctx, p)?;
            ctx.space()?;
            ctx.space()?;
            ctx.lit("=")?;
            ctx.space()?;
            write_expr(ctx, e)?;
        }
        QOrder(e, o) => {
            ctx.keyword("order")?;
            ctx.space()?;
            write_expr(ctx, e)?;
            ctx.space()?;
            write_order(ctx, o)?;
        }
    }
    Ok(())
}

pub fn write_order(ctx: &mut Context<impl Write>, o: &Order) -> Result<()> {
    match o {
        OAsc => {}
        ODesc => ctx.keyword("desc")?,
    }
    Ok(())
}

pub fn write_block(ctx: &mut Context<impl Write>, b: &Block) -> Result<()> {
    ctx.brace(|ctx| {
        ctx.indented(|ctx| {
            ctx.newline_seq(&b.ss, write_stmt)?;
            ctx.then(&b.e, |ctx, e| {
                ctx.newline()?;
                write_expr(ctx, e)
            })
        })?;
        ctx.newline()
    })?;
    Ok(())
}

pub fn write_type(ctx: &mut Context<impl Write>, t: &Type) -> Result<()> {
    match t.kind.as_ref() {
        TParen(t) => {
            ctx.paren(|ctx| write_type(ctx, t))?;
        }
        TFun(ts, t) => {
            ctx.keyword("fun")?;
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
            ctx.lit(":")?;
            write_type(ctx, t)?;
        }
        TTuple(ts) => {
            ctx.paren(|ctx| ctx.seq(ts, write_type))?;
        }
        TRecord(xts) => {
            write_type_record(ctx, xts)?;
        }
        TName(x, ts) => {
            ctx.ty(x)?;
            write_types(ctx, ts)?;
        }
        TArray(t, n) => {
            ctx.brack(|ctx| {
                write_type(ctx, t)?;
                ctx.then(n, |ctx, n| {
                    ctx.lit(";")?;
                    ctx.fmt(format_args!("{n}"))
                })
            })?;
        }
        TUnit => {
            ctx.lit("()")?;
        }
        TNever => {
            ctx.lit("!")?;
        }
        TIgnore => {
            ctx.lit("_")?;
        }
        TError => {
            ctx.lit("<Error>")?;
        }
        TRecordConcat(t0, t1) => {
            ctx.brack(|ctx| {
                write_type(ctx, t0)?;
                ctx.space()?;
                ctx.lit("&")?;
                ctx.space()?;
                write_type(ctx, t1)
            })?;
        }
        TArrayConcat(t0, t1) => {
            ctx.brack(|ctx| {
                write_type(ctx, t0)?;
                ctx.space()?;
                ctx.lit("++")?;
                ctx.space()?;
                write_type(ctx, t1)
            })?;
        }
    }
    Ok(())
}

pub fn write_unop(ctx: &mut Context<impl Write>, op: &Unop) -> Result<()> {
    ctx.fmt(format_args!("{}", &op.token))
}

pub fn write_binop(ctx: &mut Context<impl Write>, op: &Binop) -> Result<()> {
    ctx.fmt(format_args!("{}", &op.token))
}
