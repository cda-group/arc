#[path = "../../../../pretty.rs"]
pub(crate) mod pretty;

use pretty::AsPretty;
use pretty::Pretty;

use crate::compiler::ast::lower::source::lexer::Token;
use crate::compiler::info;

use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;

use std::fmt::{self, Display, Formatter};

/// State which is necessary for pretty printing tokens.
#[derive(New, Copy, Clone, Shrinkwrap)]
pub(crate) struct State<'i> {
    /// Info is needed to resolve symbols.
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i info::Info,
}

impl Token {
    /// Wraps the token inside a which can be pretty-printed.
    pub(crate) fn pretty<'i, 'j>(&'i self, info: &'j info::Info) -> Pretty<'i, Self, State<'j>> {
        self.to_pretty(State::new(info))
    }
}

impl<'i> Display for Pretty<'i, Token, State<'_>> {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let token = self.node;
        let fmt = self.fmt;
        write!(f, "\"")?;
        match token {
            Token::Indent     => write!(f, "[Indent]"),
            Token::Dedent     => write!(f, "[Dedent]"),
//=============================================================================
// Grouping
//=============================================================================
            Token::BraceL     => write!(f, "{{"),
            Token::BraceR     => write!(f, "}}"),
            Token::BraceLR    => write!(f, "{{}}"),
            Token::BrackL     => write!(f, "["),
            Token::BrackR     => write!(f, "]"),
            Token::BrackLR    => write!(f, "[]"),
            Token::ParenL     => write!(f, "("),
            Token::ParenR     => write!(f, ")"),
            Token::ParenLR    => write!(f, "()"),
            Token::AngleL     => write!(f, "<"),
            Token::AngleR     => write!(f, ">"),
            Token::AngleLR    => write!(f, "<>"),
//=============================================================================
// Operators
//=============================================================================
            Token::ArrowR     => write!(f, "->"),
            Token::AtSign     => write!(f, "@"),
            Token::Bar        => write!(f, "|"),
            Token::BarBar     => write!(f, "||"),
            Token::Colon      => write!(f, ":"),
            Token::ColonColon => write!(f, "::"),
            Token::Comma      => write!(f, ","),
            Token::Dot        => write!(f, "."),
            Token::DotDot     => write!(f, ".."),
            Token::DotDotEq   => write!(f, "..="),
            Token::Equ        => write!(f, "="),
            Token::EquEqu     => write!(f, "=="),
            Token::Geq        => write!(f, ">="),
            Token::Imply      => write!(f, "=>"),
            Token::Leq        => write!(f, "<="),
            Token::Minus      => write!(f, "-"),
            Token::Neq        => write!(f, "!="),
            Token::Percent    => write!(f, "%"),
            Token::Pipe       => write!(f, "|"),
            Token::Plus       => write!(f, "+"),
            Token::Semi       => write!(f, ";"),
            Token::Slash      => write!(f, "/"),
            Token::Star       => write!(f, "*"),
            Token::StarStar   => write!(f, "**"),
            Token::Tilde      => write!(f, "~"),
            Token::Underscore => write!(f, "_"),
//=============================================================================
// Unused Keywords
//=============================================================================
            // Token::Amp        => write!(f, "&"),
            // Token::AmpAmp     => write!(f, "&&"),
            // Token::ArrowL     => write!(f, "<-"),
            // Token::Bang       => write!(f, "!"),
            // Token::Caret      => write!(f, "^"),
            // Token::Dollar     => write!(f, "$"),
            // Token::Qm         => write!(f, "?"),
            // Token::QmQmQm     => write!(f, "???"),
            // Token::SemiSemi   => write!(f, ";;"),
//=============================================================================
// Keywords
//=============================================================================
            Token::After      => write!(f, "after"),
            Token::And        => write!(f, "and"),
            Token::As         => write!(f, "as"),
            Token::Band       => write!(f, "band"),
            Token::Bor        => write!(f, "bor"),
            Token::Break      => write!(f, "break"),
            Token::By         => write!(f, "by"),
            Token::Continue   => write!(f, "continue"),
            Token::Crate      => write!(f, "crate"),
            Token::Bxor       => write!(f, "bxor"),
            Token::Else       => write!(f, "else"),
            Token::Enwrap     => write!(f, "enwrap"),
            Token::Emit       => write!(f, "emit"),
            Token::Every      => write!(f, "every"),
            Token::Extern     => write!(f, "extern"),
            Token::Unwrap     => write!(f, "unwrap"),
            Token::Is         => write!(f, "is"),
            Token::Enum       => write!(f, "enum"),
            Token::For        => write!(f, "for"),
            Token::Fun        => write!(f, "fun"),
            Token::If         => write!(f, "if"),
            Token::In         => write!(f, "in"),
            Token::Let        => write!(f, "let"),
            Token::Log        => write!(f, "log"),
            Token::Loop       => write!(f, "loop"),
            Token::Match      => write!(f, "match"),
            Token::Not        => write!(f, "not"),
            Token::On         => write!(f, "on"),
            Token::Or         => write!(f, "or"),
            Token::Return     => write!(f, "return"),
            Token::Task       => write!(f, "task"),
            Token::Val        => write!(f, "val"),
            Token::Var        => write!(f, "var"),
            Token::Type       => write!(f, "type"),
            Token::Use        => write!(f, "use"),
            Token::Xor        => write!(f, "xor"),
//=============================================================================
// Unused Keywords
//=============================================================================
            // Token::Add        => write!(f, "add"),
            // Token::Box        => write!(f, "box"),
            // Token::Del        => write!(f, "del"),
            // Token::Do         => write!(f, "do"),
            // Token::End        => write!(f, "end"),
            // Token::Of         => write!(f, "of"),
            // Token::Port       => write!(f, "port"),
            // Token::Pub        => write!(f, "pub"),
            // Token::Reduce     => write!(f, "reduce"),
            // Token::Shutdown   => write!(f, "shutdown"),
            // Token::Sink       => write!(f, "sink"),
            // Token::Source     => write!(f, "source"),
            // Token::Startup    => write!(f, "startup"),
            // Token::State      => write!(f, "state"),
            // Token::Then       => write!(f, "then"),
            // Token::Timeout    => write!(f, "timeout"),
            // Token::Timer      => write!(f, "timer"),
            // Token::Trigger    => write!(f, "trigger"),
            // Token::Where      => write!(f, "where"),
//=============================================================================
// Primitive Types
//=============================================================================
            Token::Bool       => write!(f, "bool"),
            Token::Bf16       => write!(f, "bf16"),
            Token::F16        => write!(f, "f16"),
            Token::F32        => write!(f, "f32"),
            Token::F64        => write!(f, "f64"),
            Token::I8         => write!(f, "i8"),
            Token::I16        => write!(f, "i16"),
            Token::I32        => write!(f, "i32"),
            Token::I64        => write!(f, "i64"),
            Token::U8         => write!(f, "u8"),
            Token::U16        => write!(f, "u16"),
            Token::U32        => write!(f, "u32"),
            Token::U64        => write!(f, "u64"),
            Token::Str        => write!(f, "str"),
            Token::Unit       => write!(f, "unit"),
//=============================================================================
// Identifiers and Literals
//=============================================================================
            Token::NameId(v)  => write!(f, "{}", fmt.names.resolve(*v)),
            Token::LitI8(v)   => write!(f, "{}", v),
            Token::LitI16(v)  => write!(f, "{}", v),
            Token::LitI32(v)  => write!(f, "{}", v),
            Token::LitI64(v)  => write!(f, "{}", v),
            Token::LitU8(v)   => write!(f, "{}", v),
            Token::LitU16(v)  => write!(f, "{}", v),
            Token::LitU32(v)  => write!(f, "{}", v),
            Token::LitU64(v)  => write!(f, "{}", v),
            Token::LitBf16(v) => write!(f, "{}", v),
            Token::LitF16(v)  => write!(f, "{}", v),
            Token::LitF32(v)  => write!(f, "{}", v),
            Token::LitF64(v)  => write!(f, "{}", v),
            Token::LitBool(v) => write!(f, "{}", v),
            Token::LitChar(v) => write!(f, "{}", v),
            Token::LitStr(v)  => write!(f, "{}", v),
            Token::LitDateTime(v) => write!(f, "{:?}", v),
            Token::LitDuration(v) => write!(f, "{:?}", v),
        }?;
        write!(f, "\"")?;
        Ok(())
    }
}
