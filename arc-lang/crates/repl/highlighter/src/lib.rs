#![allow(unused)]
use std::borrow::Cow;
use std::fmt::Display;

use colored::Colorize;
use colors::BUILTIN_COLOR;
use colors::COMMENT_COLOR;
use colors::KEYWORD_COLOR;
use colors::NUMERIC_COLOR;
use colors::STRING_COLOR;
use lexer::tokens::BUILTINS;
use lexer::tokens::KEYWORDS;
use lexer::tokens::NUMERICS;
use lexer::tokens::STRINGS;
use regex::Regex;
use rustyline::highlight::Highlighter;

pub struct SyntaxHighlighter {
    pub regex: Regex,
}

fn join(patterns: impl IntoIterator<Item = impl Display>) -> String {
    patterns
        .into_iter()
        .map(|pattern| pattern.to_string())
        .collect::<Vec<_>>()
        .join("|")
}

fn capture_group(name: impl Display, pattern: impl Display) -> String {
    format!(r"(?P<{name}>{pattern})")
}

fn word(pattern: impl Display) -> String {
    format!(r"\b(?:{pattern})\b")
}

fn followed_by(a: impl Display, b: impl Display) -> String {
    format!(r"{a} *{b}")
}

fn maybe(a: impl Display) -> String {
    format!(r"(?:{a})?")
}

impl SyntaxHighlighter {
    pub fn new() -> Self {
        let pattern = &join(&[
            capture_group("keyword", &word(join(KEYWORDS))),
            capture_group("numeric", &word(join(NUMERICS))),
            capture_group("string", &join(STRINGS)),
            capture_group("builtin", &word(join(BUILTINS))),
            capture_group("comment", r"#.*"),
        ]);
        Self {
            regex: Regex::new(pattern).unwrap(),
        }
    }
}

impl Highlighter for SyntaxHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        self.regex.replace_all(line, |caps: &regex::Captures| {
            if let Some(s) = caps.name("keyword") {
                s.as_str().color(KEYWORD_COLOR).bold().to_string()
            } else if let Some(s) = caps.name("numeric") {
                s.as_str().color(NUMERIC_COLOR).to_string()
            } else if let Some(s) = caps.name("string") {
                s.as_str().color(STRING_COLOR).to_string()
            } else if let Some(s) = caps.name("builtin") {
                s.as_str().color(BUILTIN_COLOR).bold().to_string()
            } else if let Some(s) = caps.name("comment") {
                s.as_str().color(COMMENT_COLOR).to_string()
            } else {
                unreachable!()
            }
        })
    }

    fn highlight_char(&self, line: &str, _pos: usize) -> bool {
        self.regex.is_match(line)
    }
}
