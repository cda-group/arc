use colored::Color::Blue;
use colored::Colorize;
use rustyline::validate::ValidationContext;
use rustyline::validate::ValidationResult;
use rustyline::validate::Validator;
use rustyline::Result;

#[derive(Default)]
pub struct StatementValidator;

impl StatementValidator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Validator for StatementValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> Result<ValidationResult> {
        let input = ctx.input();
        let stmts = StatementIterator::new(input);
        if (stmts.count() > 0 && input.ends_with(";")) || input.ends_with("\n") {
            Ok(ValidationResult::Valid(None))
        } else {
            if input.is_empty() {
                Ok(ValidationResult::Invalid(Some(
                    "    Enter statement".color(Blue).to_string(),
                )))
            } else {
                return Ok(ValidationResult::Incomplete);
            }
        }
    }
}

pub struct StatementIterator<'a> {
    input: &'a str,
}

impl<'a> StatementIterator<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.trim(),
        }
    }
}

impl<'a> Iterator for StatementIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let mut depth = 0;
        let mut end = 0;
        for (i, c) in self.input.char_indices() {
            match c {
                '(' | '{' | '[' => depth += 1,
                ')' | '}' | ']' => depth -= 1,
                ';' if depth == 0 => {
                    end = i;
                    break;
                }
                _ => continue,
            }
        }
        if end == 0 {
            None
        } else {
            let stmt = &self.input[..end + 1].trim();
            self.input = &self.input[end + 1..];
            Some(stmt)
        }
    }
}
