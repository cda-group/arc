use std::borrow::Cow;

use colored::Color;
use colored::Color::Green;
use colored::Colorize;
use rustyline::completion::Completer;
use rustyline::completion::FilenameCompleter;
use rustyline::completion::Pair;
use rustyline::config::Configurer;
use rustyline::highlight::Highlighter;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::hint::Hinter;
use rustyline::hint::HistoryHinter;
use rustyline::history::FileHistory;
use rustyline::validate::MatchingBracketValidator;
use rustyline::validate::ValidationContext;
use rustyline::validate::ValidationResult;
use rustyline::validate::ValidationResult::Incomplete;
use rustyline::validate::ValidationResult::Invalid;
use rustyline::validate::ValidationResult::Valid;
use rustyline::validate::Validator;
use rustyline::Cmd;
use rustyline::CompletionType;
use rustyline::EditMode;
use rustyline::Editor;
use rustyline::Helper;
use rustyline::KeyCode;
use rustyline::KeyEvent;
use rustyline::Modifiers;
use rustyline::Result;

use completer::KeywordCompleter;
use highlighter::SyntaxHighlighter;
use validator::StatementValidator;

pub(crate) struct Repl {
    validator1: MatchingBracketValidator,
    validator2: StatementValidator,
    completer1: FilenameCompleter,
    completer2: KeywordCompleter,
    highlighter1: SyntaxHighlighter,
    highlighter2: MatchingBracketHighlighter,
    hinter: HistoryHinter,
    pub(crate) prompt_color: Color,
}

impl Helper for Repl {}

impl Default for Repl {
    fn default() -> Self {
        Self {
            validator1: MatchingBracketValidator::new(),
            validator2: StatementValidator::new(),
            completer1: FilenameCompleter::new(),
            completer2: KeywordCompleter::new(),
            highlighter1: SyntaxHighlighter::new(),
            highlighter2: MatchingBracketHighlighter::new(),
            hinter: HistoryHinter {},
            prompt_color: Green,
        }
    }
}

impl Repl {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl Completer for Repl {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Self::Candidate>)> {
        let (start, matches) = self.completer1.complete(line, pos, ctx)?;
        if !matches.is_empty() {
            Ok((start, matches))
        } else {
            self.completer2.complete(line, pos, ctx)
        }
    }
}

impl Validator for Repl {
    fn validate(&self, ctx: &mut ValidationContext) -> Result<ValidationResult> {
        if let r @ (Incomplete | Invalid(_)) = self.validator1.validate(ctx)? {
            return Ok(r);
        }
        if let r @ (Incomplete | Invalid(_)) = self.validator2.validate(ctx)? {
            return Ok(r);
        }
        Ok(Valid(None))
    }
}

impl Highlighter for Repl {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        let line = self.highlighter1.highlight(line, pos);
        match line {
            Cow::Borrowed(line) => self.highlighter2.highlight(line, pos),
            Cow::Owned(line) => match self.highlighter2.highlight(line.as_str(), pos) {
                Cow::Borrowed(line) => Cow::Owned(line.to_string()),
                Cow::Owned(line) => Cow::Owned(line),
            },
        }
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            Cow::Owned(prompt.color(self.prompt_color).to_string())
        } else {
            Cow::Borrowed(prompt)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Borrowed(hint)
    }

    fn highlight_candidate<'c>(
        &self,
        candidate: &'c str, // FIXME should be Completer::Candidate
        completion: CompletionType,
    ) -> Cow<'c, str> {
        let _ = completion;
        Cow::Borrowed(candidate)
    }

    fn highlight_char(&self, line: &str, pos: usize) -> bool {
        self.highlighter1.highlight_char(line, pos) || self.highlighter2.highlight_char(line, pos)
    }
}

impl Hinter for Repl {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        self.hinter.hint(line, pos, ctx)
    }
}
