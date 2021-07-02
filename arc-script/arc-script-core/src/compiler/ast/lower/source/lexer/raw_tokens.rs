//! Module which generates a `LogosLexer`.

use logos::Lexer as LogosLexer;
use logos::Logos;

/// A raw token which carries no semantic information.
#[rustfmt::skip]
#[derive(Logos, Debug, PartialEq)]
#[logos(subpattern bin = r"0[bB][0-1][_0-1]*")]
#[logos(subpattern oct = r"0[oB][0-7][_0-7]*")]
#[logos(subpattern dec = r"[0-9][_0-9]*")]
#[logos(subpattern hex = r"0[xX][0-9a-fA-F][_0-9a-fA-F]*")]
#[logos(subpattern exp = r"[eE][+-]?[0-9][_0-9]*")]
pub(crate) enum LogosToken {
    #[regex(r"[ \t\f\n]+", logos::skip)]
    #[error]
    Error,
    #[regex(r"#[^r\n]*", priority = 2)]  // # ...
    #[regex(r"#[^\n]*", priority = 1)]   // # ...
    #[regex(r"#\*([^*]|\*+[^*#])*\*+#")] // #* ... *#
    Comment,
    #[regex(r"([\n\t\f] *)+", priority = 2)]
    Newline,
//=============================================================================
// Grouping
//=============================================================================
    #[token("(")] ParenL,
    #[token(")")] ParenR,
    #[token("()")] ParenLR,
    #[token("[")] BrackL,
    #[token("]")] BrackR,
    #[token("[]")] BrackLR,
    #[token("{")] BraceL,
    #[token("}")] BraceR,
    #[token("{}")] BraceLR,
    #[token("<")] AngleL,
    #[token(">")] AngleR,
    #[token("<>")] AngleLR,
//=============================================================================
// Operators
//=============================================================================
    #[token("!=")] Neq,
    #[token("%")] Percent,
    #[token("*")] Star,
    #[token("**")] StarStar,
    #[token("+")] Plus,
    #[token(",")] Comma,
    #[token("-")] Minus,
    #[token("->")] ArrowR,
    #[token(".")] Dot,
    #[token("..")] DotDot,
    #[token("..=")] DotDotEq,
    #[token("/")] Slash,
    #[token(":")] Colon,
    #[token("::")] ColonColon,
    #[token(";")] Semi,
    #[token("<=")] Leq,
    #[token("=")] Equ,
    #[token("==")] EquEqu,
    #[token("=>")] Imply,
    #[token(">=")] Geq,
    #[token("@")] AtSign,
    #[token("_")] Underscore,
    #[token("|")] Bar,
    #[token("||")] BarBar,
    #[token("~")] Tilde,
//=============================================================================
// Retired Operators
//=============================================================================
    // #[token("|>")] Pipe,
    // #[token("!")] Bang,
    // #[token("$")] Dollar,
    // #[token("&")] Amp,
    // #[token("&&")] AmpAmp,
    // #[token(";;")] SemiSemi,
    // #[token("<-")] ArrowL,
    // #[token("?")] Qm,
    // #[token("???")] QmQmQm,
    // #[token("^")] Caret,
//=============================================================================
// Keywords
//=============================================================================
    #[token("after")] After,
    #[token("and")] And,
    #[token("as")] As,
    #[token("break")] Break,
    #[token("band")] Band,
    #[token("bor")] Bor,
    #[token("bxor")] Bxor,
    #[token("by")] By,
    #[token("crate")] Crate,
    #[token("continue")] Continue,
    #[token("else")] Else,
    #[token("emit")] Emit,
    #[token("enum")] Enum,
    #[token("every")] Every,
    #[token("extern")] Extern,
    #[token("for")] For,
    #[token("fun")] Fun,
    #[token("if")] If,
    #[token("in")] In,
    #[token("is")] Is,
    #[token("let")] Let,
    #[token("log")] Log,
    #[token("loop")] Loop,
    #[token("match")] Match,
    #[token("not")] Not,
    #[token("on")] On,
    #[token("or")] Or,
    #[token("return")] Return,
    #[token("task")] Task,
    #[token("type")] Type,
    #[token("val")] Val,
    #[token("var")] Var,
    #[token("unwrap")] Unwrap,
    #[token("enwrap")] Enwrap,
    #[token("use")] Use,
    #[token("xor")] Xor,
//=============================================================================
// Reserved Keywords
//=============================================================================
    // #[token("add")] Add,
    // #[token("box")] Box,
    // #[token("do")] Do,
    // #[token("end")] End,
    // #[token("of")] Of,
    // #[token("port")] Port,
    // #[token("pub")] Pub,
    // #[token("reduce")] Reduce,
    // #[token("shutdown")] Shutdown,
    // #[token("sink")] Sink,
    // #[token("source")] Source,
    // #[token("startup")] Startup,
    // #[token("state")] State,
    // #[token("then")] Then,
    // #[token("timeout")] Timeout,
    // #[token("timer")] Timer,
    // #[token("trigger")] Trigger,
    // #[token("where")] Where,
//=============================================================================
// Primitive Types
//=============================================================================
    #[token("bool")] Bool,
    #[token("f32")] F32,
    #[token("f64")] F64,
    #[token("i8")] I8,
    #[token("i16")] I16,
    #[token("i32")] I32,
    #[token("i64")] I64,
    #[token("u8")] U8,
    #[token("u16")] U16,
    #[token("u32")] U32,
    #[token("u64")] U64,
    #[token("str")] Str,
    #[token("unit")] Unit,
//=============================================================================
// Identifiers and Literals
//=============================================================================
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*'*")]                            NameId,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))i8")]               LitI8,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))i16")]              LitI16,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))")]                 LitI32,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))i64")]              LitI64,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))u8")]               LitU8,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))u16")]              LitU16,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))u32")]              LitU32,
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))u64")]              LitU64,
    #[regex(r"-?(((?&dec)\.(?&dec)(?&exp)?)|((?&dec)(?&exp)))f32")]  LitF32,
    #[regex(r"-?(((?&dec)\.(?&dec)(?&exp)?)|((?&dec)(?&exp)))")]     LitF64,
    #[token("true")]         LitTrue,
    #[token("false")]        LitFalse,
    #[regex(r"'[^']'")]      LitChar,
    #[regex(r#""[^"]*""#)]   LitStr,
    #[regex(r"(?&dec)+ns")]  LitDurationNs,
    #[regex(r"(?&dec)+us")]  LitDurationUs,
    #[regex(r"(?&dec)+ms")]  LitDurationMs,
    #[regex(r"(?&dec)+s")]   LitDurationS,
    #[regex(r"(?&dec)+m")]   LitDurationM,
    #[regex(r"(?&dec)+h")]   LitDurationH,
    #[regex(r"(?&dec)+d")]   LitDurationD,
    #[regex(r"(?&dec)+w")]   LitDurationW,
    // #[regex(r"[0-9]+mo")]  LitDurationMo,
    // #[regex(r"[0-9]+y")]   LitDurationY,
    #[regex(r"(?&dec)-(?&dec)-(?&dec)")]                                              LitDate,
    #[regex(r"(?&dec)-(?&dec)-(?&dec)T(?&dec):(?&dec):(?&dec)")]                      LitDateTime,
    #[regex(r"(?&dec)-(?&dec)-(?&dec)T(?&dec):(?&dec):(?&dec)(\+|-)(?&dec):(?&dec)")] LitDateTimeZone,
}
