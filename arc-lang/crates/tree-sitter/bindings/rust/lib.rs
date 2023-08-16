//! This crate provides arc-lang language support for the [tree-sitter][] parsing library.
//!
//! Typically, you will use the [language][language func] function to add this language to a
//! tree-sitter [Parser][], and then use the parser to parse some code:
//!
//! [Language]: https://docs.rs/tree-sitter/*/tree_sitter/struct.Language.html
//! [language func]: fn.language.html
//! [Parser]: https://docs.rs/tree-sitter/*/tree_sitter/struct.Parser.html
//! [tree-sitter]: https://tree-sitter.github.io/

use tree_sitter::Language;

extern "C" {
    fn tree_sitter_arc_lang() -> Language;
}

/// Get the tree-sitter [Language][] for this grammar.
///
/// [Language]: https://docs.rs/tree-sitter/*/tree_sitter/struct.Language.html
pub fn language() -> Language {
    unsafe { tree_sitter_arc_lang() }
}

/// The content of the [`node-types.json`][] file for this grammar.
///
/// [`node-types.json`]: https://tree-sitter.github.io/tree-sitter/using-parsers#static-node-types
pub const NODE_TYPES: &str = include_str!("../../src/node-types.json");

// Uncomment these to include any queries that this grammar contains

pub const HIGHLIGHTS_QUERY: &str = include_str!("../../queries/arc_lang/highlights.scm");
pub const INJECTIONS_QUERY: &str = include_str!("../../queries/arc_lang/injections.scm");
pub const LOCALS_QUERY: &str = include_str!("../../queries/arc_lang/locals.scm");
pub const TAGS_QUERY: &str = include_str!("../../queries/arc_lang/tags.scm");
pub const HIGHLIGHT_NAMES: &[&str] = &[
    "string",
    "number",
    "boolean",
    "comment",
    "keyword",
    "function",
    "variable",
    "type",
    "type.builtin",
    "conditional",
    "repeat",
    "punctuation",
    "operator",
];

#[cfg(test)]
mod tests {
    use tree_sitter_highlight::HighlightConfiguration;
    use tree_sitter_highlight::Highlighter;

    #[test]
    fn can_highlight() {
        let code = "def foo() = 1+1;";

        let mut highlighter = Highlighter::new();
        let mut config = HighlightConfiguration::new(
            super::language(),
            super::HIGHLIGHTS_QUERY,
            super::INJECTIONS_QUERY,
            super::LOCALS_QUERY,
        )
        .unwrap();

        config.configure(super::HIGHLIGHT_NAMES);

        use tree_sitter_highlight::HighlightEvent;

        let highlights = highlighter
            .highlight(&config, code.as_bytes(), None, |_| None)
            .unwrap();

        for event in highlights {
            if let Ok(event) = event {
                match event {
                    HighlightEvent::Source { start, end } => {
                        eprint!("[{}]", &code[start..end]);
                    }
                    HighlightEvent::HighlightStart(i) => {
                        eprint!("<{}", i.0);
                    }
                    HighlightEvent::HighlightEnd => {
                        eprint!(">");
                    }
                }
            }
        }
    }
}
