//! Module which generates a `LogosLexer`.

use diagnostics::LexerError;
use std::fmt::Debug;
use std::fmt::Display;
use std::str::FromStr;

use logos::Lexer;
use logos::Logos;

#[rustfmt::skip]
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(error = LexerError)]
#[logos(subpattern bin = r"0[bB][0-1][_0-1]*")]
#[logos(subpattern oct = r"0[oO][0-7][_0-7]*")]
#[logos(subpattern dec = r"[0-9][_0-9]*")]
#[logos(subpattern hex = r"0[xX][0-9a-fA-F][_0-9a-fA-F]*")]
#[logos(subpattern exp = r"[eE][+-]?[0-9][_0-9]*")]
#[logos(subpattern id = r"[A-Za-z_][A-Za-z0-9_]*")]
#[logos(skip r"([ \t\f\n]+|#[^\r\n]*|#[^\n]*)")]
pub enum Token {
    #[token("(")] ParenL,
    #[token(")")] ParenR,
    #[token("[")] BrackL,
    #[token("]")] BrackR,
    #[token("{")] BraceL,
    #[token("}")] BraceR,
    #[token("<")] AngleL,
    #[token(">")] AngleR,
    // Operators
    #[token("!=")] Neq,
    #[token("*")] Star,
    #[token("+")] Plus,
    #[token("++")] PlusPlus,
    #[token("&")] Ampersand,
    #[token(",")] Comma,
    #[token("-")] Minus,
    #[token(".")] Dot,
    #[token("..")] DotDot,
    #[token("..=")] DotDotEq,
    #[token("/")] Slash,
    #[token(":")] Colon,
    #[token("::")] ColonColon,
    #[token(";")] Semi,
    #[token("<=")] Leq,
    #[token("=")] Eq,
    #[token("==")] EqEq,
    #[token("=>")] Imply,
    #[token(">=")] Geq,
    #[token("@")] AtSign,
    #[token("_")] Underscore,
    #[token("|")] Bar,
    #[token("+=")] PlusEq,
    #[token("-=")] MinusEq,
    #[token("*=")] StarEq,
    #[token("/=")] SlashEq,
    #[token("~")] Tilde,
    #[token("!")] Bang,
    #[token("?")] Question,
    // Keywords
    #[token("and")] And,
    #[token("as")] As,
    #[token("break")] Break,
    #[token("catch")] Catch,
    #[token("continue")] Continue,
    #[token("def")] Def,
    #[token("desc")] Desc,
    #[token("do")] Do,
    #[token("else")] Else,
    #[token("finally")] Finally,
    #[token("for")] For,
    #[token("from")] From,
    #[token("fun")] Fun,
    #[token("group")] Group,
    #[token("if")] If,
    #[token("in")] In,
    #[token("into")] Into,
    #[token("join")] Join,
    #[token("loop")] Loop,
    #[token("match")] Match,
    #[token("not")] Not,
    #[token("on")] On,
    #[token("or")] Or,
    #[token("of")] Of,
    #[token("return")] Return,
    #[token("select")] Select,
    #[token("compute")] Compute,
    #[token("repeat")] Repeat,
    #[token("throw")] Throw,
    #[token("try")] Try,
    #[token("type")] Type,
    #[token("val")] Val,
    #[token("var")] Var,
    #[token("with")] With,
    #[token("where")] Where,
    #[token("while")] While,
    #[token("use")] Use,
    #[token("union")] Union,
    #[token("over")] Over,
    #[token("roll")] Roll,
    #[token("order")] Order,
    #[token("enum")] Enum,
    #[token("---")] MinusMinusMinus,
    #[logos(skip)] Inject((String, String)),
    // Identifiers and Literals
    #[regex(r"(?&id)", |lex| lex.slice().to_string())] Name(String),
    // TODO: Parse without -
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))", |lex| lex.slice().parse())] Int(i32),
    #[regex(r"-?((?&bin)|(?&dec)|(?&hex)|(?&oct))(?&id)", Token::parse_suffix)] IntSuffix((i32, String)),
    #[regex(r"-?(((?&dec)\.(?&dec)(?&exp)?)|((?&dec)(?&exp)))", |lex| lex.slice().parse())] Float(f32),
    #[regex(r"-?(((?&dec)\.(?&dec)(?&exp)?)|((?&dec)(?&exp)))(?&id)", Token::parse_suffix)] FloatSuffix((f32, String)),

    #[regex(r#"true|false"#, |lex| lex.slice().parse())] Bool(bool),
    #[regex(r"'[^']'", |lex| lex.slice().get(1..lex.slice().len()-1).unwrap().parse())] Char(char),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice().get(1..lex.slice().len()-1).unwrap_or("").to_string())] String(String),
}

impl Token {
    fn parse_suffix<'s, T>(lexer: &Lexer<'s, Token>) -> Result<(T, String), <T as FromStr>::Err>
    where
        T: std::str::FromStr,
        <T as FromStr>::Err: Debug,
    {
        let slice = lexer.slice();
        let end = slice.chars().position(|char| !char.is_digit(10)).unwrap();
        let int = slice[..end].parse()?;
        let suffix = slice[end..].to_string();
        Ok((int, suffix))
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::ParenL => write!(f, "("),
            Token::ParenR => write!(f, ")"),
            Token::BrackL => write!(f, "["),
            Token::BrackR => write!(f, "]"),
            Token::BraceL => write!(f, "{{"),
            Token::BraceR => write!(f, "}}"),
            Token::AngleL => write!(f, "<"),
            Token::AngleR => write!(f, ">"),
            Token::Neq => write!(f, "!="),
            Token::Star => write!(f, "*"),
            Token::Plus => write!(f, "+"),
            Token::PlusPlus => write!(f, "++"),
            Token::Comma => write!(f, ","),
            Token::Minus => write!(f, "-"),
            Token::Dot => write!(f, "."),
            Token::DotDot => write!(f, ".."),
            Token::DotDotEq => write!(f, "..="),
            Token::Slash => write!(f, "/"),
            Token::Colon => write!(f, ":"),
            Token::ColonColon => write!(f, "::"),
            Token::Semi => write!(f, ";"),
            Token::Leq => write!(f, "<="),
            Token::Eq => write!(f, "="),
            Token::EqEq => write!(f, "=="),
            Token::Imply => write!(f, "=>"),
            Token::Geq => write!(f, ">="),
            Token::Ampersand => write!(f, "&"),
            Token::AtSign => write!(f, "@"),
            Token::Underscore => write!(f, "_"),
            Token::Bar => write!(f, "|"),
            Token::PlusEq => write!(f, "+="),
            Token::MinusEq => write!(f, "-="),
            Token::StarEq => write!(f, "*="),
            Token::SlashEq => write!(f, "/="),
            Token::Tilde => write!(f, "~"),
            Token::Bang => write!(f, "!"),
            Token::Question => write!(f, "?"),
            Token::And => write!(f, "and"),
            Token::As => write!(f, "as"),
            Token::Break => write!(f, "break"),
            Token::Catch => write!(f, "catch"),
            Token::Continue => write!(f, "continue"),
            Token::Def => write!(f, "def"),
            Token::Desc => write!(f, "desc"),
            Token::Do => write!(f, "do"),
            Token::Else => write!(f, "else"),
            Token::Finally => write!(f, "finally"),
            Token::For => write!(f, "for"),
            Token::From => write!(f, "from"),
            Token::Fun => write!(f, "fun"),
            Token::Group => write!(f, "group"),
            Token::If => write!(f, "if"),
            Token::In => write!(f, "in"),
            Token::Into => write!(f, "into"),
            Token::Join => write!(f, "join"),
            Token::Loop => write!(f, "loop"),
            Token::Match => write!(f, "match"),
            Token::Not => write!(f, "not"),
            Token::On => write!(f, "on"),
            Token::Or => write!(f, "or"),
            Token::Of => write!(f, "of"),
            Token::Return => write!(f, "return"),
            Token::Select => write!(f, "select"),
            Token::Compute => write!(f, "compute"),
            Token::Repeat => write!(f, "repeat"),
            Token::Throw => write!(f, "throw"),
            Token::Try => write!(f, "try"),
            Token::Type => write!(f, "type"),
            Token::Val => write!(f, "val"),
            Token::Var => write!(f, "var"),
            Token::With => write!(f, "with"),
            Token::Where => write!(f, "where"),
            Token::While => write!(f, "while"),
            Token::Use => write!(f, "use"),
            Token::Union => write!(f, "union"),
            Token::Name(x) => write!(f, "{x}"),
            Token::Int(l) => write!(f, "{l}"),
            Token::IntSuffix((l, x)) => write!(f, "{l}{x}"),
            Token::Float(l) => write!(f, "{l}"),
            Token::FloatSuffix((l, x)) => write!(f, "{l}{x}"),
            Token::Bool(l) => write!(f, "{l}"),
            Token::Char(l) => write!(f, "{l}"),
            Token::String(l) => write!(f, "{l}"),
            Token::Over => write!(f, "over"),
            Token::Roll => write!(f, "roll"),
            Token::Order => write!(f, "order"),
            Token::Enum => write!(f, "enum"),
            Token::MinusMinusMinus => unreachable!(),
            Token::Inject((lang, code)) => write!(f, "---{lang}{code}---"),
        }
    }
}

pub const KEYWORDS: &[&str] = &[
    "and", "as", "break", "catch", "continue", "def", "desc", "do", "else", "finally", "for",
    "from", "fun", "group", "if", "in", "into", "join", "loop", "match", "not", "on", "or", "of",
    "return", "select", "compute", "repeat", "throw", "try", "type", "val", "var", "where", "with",
    "while", "use", "union", "over", "roll", "order", "enum", "rust",
];

pub const NUMERICS: &[&str] = &[
    "0[bB][0-1][_0-1]*",
    "0[oO][0-7][_0-7]*",
    "[0-9][_0-9]*",
    "0[xX][0-9a-fA-F][_0-9a-fA-F]*",
];

pub const BUILTINS: &[&str] = &["true", "false"];

pub const STRINGS: &[&str] = &[r#""([^"\\]|\\.)*""#, r"'[^']'"];

pub const COMMENTS: &[&str] = &["#[^\r\n]*", "#[^\n]*"];

pub const TYPES: &[&str] = &[
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "f32",
    "f64",
    "Int",
    "Float",
    "bool",
    "char",
    "String",
    "Vec",
    "Option",
    "Result",
    "Set",
    "Dict",
    "Stream",
    "Matrix",
    "File",
    "SocketAddr",
    "Url",
    "Path",
    "Duration",
    "Time",
];
