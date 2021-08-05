use super::pattern::Case;
use super::Context;

use crate::ast;
use crate::hir;
use crate::hir::HIR;
use crate::info::diags::Error;
use crate::info::files::Loc;
use crate::info::types::TypeId;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;

use crate::hir::lower::ast::special::path;

#[derive(Shrinkwrap)]
pub(crate) struct CaseDebug<'a> {
    case: &'a Case,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'a Info,
    hir: &'a HIR,
}

impl Case {
    pub(crate) const fn debug<'a>(&'a self, info: &'a Info, hir: &'a HIR) -> CaseDebug<'a> {
        CaseDebug {
            case: self,
            info,
            hir,
        }
    }
}

use crate::info::Info;
use std::fmt::Display;

impl<'a> Display for CaseDebug<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CaseDebug: {{")?;
        match self.case {
            Case::Guard(cond) => writeln!(f, "    true == {}", self.hir.pretty(cond, self.info))?,
            Case::Stmt(s) => writeln!(f, "    {}", self.hir.pretty(s, self.info))?,
            _ => {}
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}
