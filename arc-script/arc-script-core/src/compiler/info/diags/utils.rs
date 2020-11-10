use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::diags::Diagnostic;
use crate::compiler::info::diags::Error;
use crate::compiler::info::diags::Note;
use crate::compiler::info::diags::Result;
use crate::compiler::info::diags::Warning;

impl Into<Diagnostic> for Warning {
    fn into(self) -> Diagnostic {
        Diagnostic::Warning(self)
    }
}

impl Into<Diagnostic> for Error {
    fn into(self) -> Diagnostic {
        Diagnostic::Error(self)
    }
}

impl Into<Diagnostic> for Note {
    fn into(self) -> Diagnostic {
        Diagnostic::Note(self)
    }
}
