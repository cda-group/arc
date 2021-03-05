//! Module which generates a `LogosLexer`.

use logos::Lexer as LogosLexer;
use logos::Logos;

/// A raw token which carries no semantic information.
#[rustfmt::skip]
#[derive(Logos, Debug, PartialEq)]
pub(crate) enum LogosToken {
    #[regex(r"[ \t\f\n]+", logos::skip)]
    #[error]
    Error,
    #[regex(r"#[^r\n]*", priority = 2)]
    #[regex(r"#[^\n]*", priority = 1)]
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
//=============================================================================
// Operators
//=============================================================================
    #[token("&")] Amp,
    #[token("&&")] AmpAmp,
    #[token("<-")] ArrowL,
    #[token("->")] ArrowR,
    #[token("@")] AtSign,
    #[token("!")] Bang,
    #[token("|")] Bar,
    #[token("||")] BarBar,
    #[token("^")] Caret,
    #[token(":")] Colon,
    #[token("::")] ColonColon,
    #[token(",")] Comma,
    #[token(".")] Dot,
    #[token("..")] DotDot,
    #[token("=")] Equ,
    #[token("==")] EquEqu,
    #[token(">=")] Geq,
    #[token(">")] Gt,
    #[token("=>")] Imply,
    #[token("<=")] Leq,
    #[token("<")] Lt,
    #[token("<>")] LtGt,
    #[token("-")] Minus,
    #[token("!=")] Neq,
    #[token("%")] Percent,
    #[token("|>")] Pipe,
    #[token("+")] Plus,
    #[token("?")] Qm,
    #[token("???")] QmQmQm,
    #[token(";")] Semi,
    #[token(";;")] SemiSemi,
    #[token("/")] Slash,
    #[token("*")] Star,
    #[token("**")] StarStar,
    #[token("~")] Tilde,
    #[token("_")] Underscore,
    #[token("$")] Dollar,
//=============================================================================
// Keywords
//=============================================================================
    #[token("after")] After,
    #[token("and")] And,
    #[token("as")] As,
    #[token("break")] Break,
    #[token("band")] Band,
    #[token("box")] Box,
    #[token("bor")] Bor,
    #[token("bxor")] Bxor,
    #[token("by")] By,
    #[token("crate")] Crate,
    #[token("else")] Else,
    #[token("emit")] Emit,
    #[token("enum")] Enum,
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
    #[token("pub")] Pub,
    #[token("reduce")] Reduce,
    #[token("return")] Return,
    #[token("state")] State,
    #[token("task")] Task,
    #[token("type")] Type,
    #[token("unwrap")] Unwrap,
    #[token("enwrap")] Enwrap,
    #[token("use")] Use,
    #[token("xor")] Xor,
//=============================================================================
// Reserved Keywords
//=============================================================================
    #[token("end")] End,
    #[token("of")] Of,
    #[token("shutdown")] Shutdown,
    #[token("sink")] Sink,
    #[token("source")] Source,
    #[token("then")] Then,
    #[token("where")] Where,
//=============================================================================
// Primitive Types
//=============================================================================
    #[token("bool")] Bool,
    #[token("bf16")] Bf16,
    #[token("f16")] F16,
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
    #[token("null")] Null,
    #[token("str")] Str,
    #[token("unit")] Unit,
//=============================================================================
// Identifiers and Literals
//=============================================================================
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*'*")] NameId,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))i8")]                   LitI8,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))i16")]                  LitI16,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))")]                     LitI32,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))i64")]                  LitI64,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))u8")]                   LitU8,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))u16")]                  LitU16,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))u32")]                  LitU32,
    #[regex(r"-?(([0-9]+)|(0b[0-1]+)|(0x[0-9a-fA-F]+))u64")]                  LitU64,
    #[regex(r"-?(([0-9]+\.[0-9]+([eE]-?[0-9]+)?)|([0-9]+[eE]-?[0-9]+))bf16")] LitBf16,
    #[regex(r"-?(([0-9]+\.[0-9]+([eE]-?[0-9]+)?)|([0-9]+[eE]-?[0-9]+))f16")]  LitF16,
    #[regex(r"-?(([0-9]+\.[0-9]+([eE]-?[0-9]+)?)|([0-9]+[eE]-?[0-9]+))f32")]  LitF32,
    #[regex(r"-?(([0-9]+\.[0-9]+([eE]-?[0-9]+)?)|([0-9]+[eE]-?[0-9]+))")]     LitF64,
    #[token("true")]       LitTrue,
    #[token("false")]      LitFalse,
    #[regex(r"'[^']'")]    LitChar,
    #[regex(r#""[^"]*""#)] LitStr,
    #[regex(r"[0-9]+s")]   LitS,
    #[regex(r"[0-9]+us")]  LitUs,
    #[regex(r"[0-9]+ms")]  LitMs,
    #[regex(r"[0-9]+ns")]  LitNs,
    #[regex(r"[0-9]+min")] LitMins,
    #[regex(r"[0-9]+h")]   LitHrs,
}
