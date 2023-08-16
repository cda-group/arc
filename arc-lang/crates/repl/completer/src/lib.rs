use lexer::tokens::KEYWORDS;
use rustyline::completion::Completer;
use rustyline::completion::Pair;
use rustyline::Result;

pub struct KeywordCompleter;

impl KeywordCompleter {
    pub fn new() -> Self {
        Self
    }
}

impl Completer for KeywordCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Self::Candidate>)> {
        let mut start = pos - 1;
        let mut end = pos;
        let chars = line.chars().collect::<Vec<_>>();
        while start > 0 {
            match chars[start] {
                'a'..='z' => start -= 1,
                _ => break,
            }
        }
        while end < line.len() {
            match chars[end] {
                'a'..='z' => end += 1,
                _ => break,
            }
        }
        let matches = KEYWORDS
            .into_iter()
            .filter(|cmd| cmd.starts_with(&line[start..end]))
            .map(|x| Pair {
                display: x.to_string(),
                replacement: x.to_string(),
            })
            .collect();
        Ok((start, matches))
    }
}
