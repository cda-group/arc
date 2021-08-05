use sqlparser::ast::ColumnDef;
use sqlparser::ast::Statement as SqlStmt;
use sqlparser::dialect::Dialect;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Query {
    stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt {
    Sql(SqlStmt),
    Arq(ArqStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArqStmt {
    StreamExternalTable {
        /// Table name
        name: String,
        /// Optional schema
        column_defs: Vec<ColumnDef>,
    },
}

#[derive(Debug)]
pub(crate) struct ArqDialect;

impl Dialect for ArqDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        ch == '"' || ch == '['
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        ('a'..='z').contains(&ch) || ('A'..='Z').contains(&ch) || ch == '_'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        ('a'..='z').contains(&ch)
            || ('A'..='Z').contains(&ch)
            || ('0'..='9').contains(&ch)
            || ch == '_'
    }
}
