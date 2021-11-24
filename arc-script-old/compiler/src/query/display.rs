use crate::query::repr::ArqStmt;
use crate::query::repr::Stmt;

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;

impl Display for Stmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Sql(stmt) => write!(f, "{}", stmt),
            Self::Arq(stmt) => write!(f, "{}", stmt),
        }
    }
}

impl Display for ArqStmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::StreamExternalTable { name, column_defs } => {
                writeln!(f, "STREAM EXTERNAL TABLE {}", name)?;
                let mut iter = column_defs.iter();
                if let Some(column_def) = iter.next() {
                    write!(f, "({}", column_def)?;
                    for column_def in column_defs {
                        write!(f, ", {}", column_def)?;
                    }
                    write!(f, ")")?;
                }
            }
        }
        Ok(())
    }
}
