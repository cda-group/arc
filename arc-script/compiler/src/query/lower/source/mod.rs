use crate::query::ArqDialect;
use crate::query::ArqStmt;
use crate::query::Stmt;

use sqlparser::ast::ColumnDef;
use sqlparser::ast::ColumnOptionDef;
use sqlparser::ast::TableConstraint;
use sqlparser::dialect::keywords::Keyword;
use sqlparser::parser::Parser as SqlParser;
use sqlparser::parser::ParserError;
use sqlparser::tokenizer::Token;
use sqlparser::tokenizer::Tokenizer;

// Use `Parser::expected` instead, if possible
macro_rules! err {
    ($MSG:expr) => {
        Err(ParserError::ParserError($MSG.to_string()))
    };
}

/// ARQ-SQL Parser
pub(crate) struct Parser<'a> {
    sql: SqlParser<'a>,
}

impl<'a> Parser<'a> {
    /// Parse the specified tokens
    pub(crate) fn new(source: &str) -> Result<Self, ParserError> {
        let dialect = &ArqDialect;
        let tokens = Tokenizer::new(dialect, source).tokenize()?;

        Ok(Parser {
            sql: SqlParser::new(tokens, dialect),
        })
    }

    /// Parse a SQL statement and produce a set of statements
    pub(crate) fn parse(source: &str) -> Result<Vec<Stmt>, ParserError> {
        let mut parser = Parser::new(source)?;
        let mut stmts = Vec::new();
        let mut semi_expected = false;
        loop {
            while parser.consume(&Token::SemiColon) {
                semi_expected = false;
            }

            if parser.peek() == Token::EOF {
                break;
            }

            if semi_expected {
                return parser.expect("end of statement");
            }

            stmts.push(parser.parse_stmt()?);
            semi_expected = true;
        }
        Ok(stmts)
    }

    /// Report unexpected token
    fn expect<T>(&mut self, expected: &str) -> Result<T, ParserError> {
        let found = self.peek();
        err!(format!("Expected {}, found: {}", expected, found))
    }

    fn peek(&mut self) -> Token {
        self.sql.peek_token()
    }

    fn consume(&mut self, expected: &Token) -> bool {
        self.sql.consume_token(expected)
    }

    fn next(&mut self) -> Token {
        self.sql.next_token()
    }

    /// Parse a new expression
    fn parse_stmt(&mut self) -> Result<Stmt, ParserError> {
        match self.peek() {
            Token::Word(w) => {
                if w.value.as_str() == "STREAM" {
                    self.next();
                    self.parse_stream()
                } else {
                    Ok(Stmt::Sql(self.sql.parse_statement()?))
                }
            }
            _ => Ok(Stmt::Sql(self.sql.parse_statement()?)),
        }
    }

    /// Parse a STREAM statement
    fn parse_stream(&mut self) -> Result<Stmt, ParserError> {
        if self.sql.parse_keyword(Keyword::EXTERNAL) {
            self.parse_stream_external_table()
        } else {
            Ok(Stmt::Sql(self.sql.parse_create()?))
        }
    }

    fn parse_column_defs(&mut self) -> Result<(Vec<ColumnDef>, Vec<TableConstraint>), ParserError> {
        let mut column_defs = vec![];
        let mut constraints = vec![];
        if !self.consume(&Token::LParen) || self.consume(&Token::RParen) {
            return Ok((column_defs, constraints));
        }
        loop {
            if let Some(constraint) = self.sql.parse_optional_table_constraint()? {
                constraints.push(constraint);
            } else if let Token::Word(_) = self.peek() {
                column_defs.push(self.parse_column_def()?);
            } else {
                return self.expect("column name or constraint definition");
            }
            let comma = self.consume(&Token::Comma);
            if self.consume(&Token::RParen) {
                break;
            } else if !comma {
                return self.expect("',' or ')' after column definition");
            }
        }

        Ok((column_defs, constraints))
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef, ParserError> {
        let name = self.sql.parse_identifier()?;
        let data_type = self.sql.parse_data_type()?;
        let collation = if self.sql.parse_keyword(Keyword::COLLATE) {
            Some(self.sql.parse_object_name()?)
        } else {
            None
        };
        let mut options = vec![];
        loop {
            if self.sql.parse_keyword(Keyword::CONSTRAINT) {
                let name = Some(self.sql.parse_identifier()?);
                if let Some(option) = self.sql.parse_optional_column_option()? {
                    options.push(ColumnOptionDef { name, option });
                } else {
                    return self.expect("constraint details after CONSTRAINT <name>");
                }
            } else if let Some(option) = self.sql.parse_optional_column_option()? {
                options.push(ColumnOptionDef { name: None, option });
            } else {
                break;
            };
        }
        Ok(ColumnDef {
            name,
            data_type,
            collation,
            options,
        })
    }

    fn parse_stream_external_table(&mut self) -> Result<Stmt, ParserError> {
        self.sql.expect_keyword(Keyword::TABLE)?;
        let table_name = self.sql.parse_object_name()?;
        let (columns, _) = self.parse_column_defs()?;

        Ok(Stmt::Arq(ArqStmt::StreamExternalTable {
            name: table_name.to_string(),
            column_defs: columns,
        }))
    }
}
