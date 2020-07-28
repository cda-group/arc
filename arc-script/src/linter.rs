use {
    crate::{
        ast::Script,
        error::{Diagnostic, SimpleFile},
    },
    codespan::ByteIndex,
    codespan_lsp::byte_index_to_position,
    codespan_reporting::diagnostic::{Label, Severity},
    tower_lsp::lsp_types as lsp,
};

impl<'i> Script<'i> {
    pub fn to_lsp(&self) -> Vec<lsp::Diagnostic> {
        let file = &SimpleFile::new("input", self.info.source);
        let mut files = codespan::Files::new();
        let id = files.add("", file.source());
        self.info
            .errors
            .iter()
            .map(|error| error.to_diagnostic(&self.info))
            .flat_map(
                |Diagnostic {
                     severity,
                     code,
                     message: source,
                     labels,
                     ..
                 }| {
                    labels
                        .into_iter()
                        .map(|Label { range, message, .. }| lsp::Diagnostic {
                            range: lsp::Range::new(
                                byte_index_to_position(&files, id, ByteIndex(range.start as u32))
                                    .unwrap(),
                                byte_index_to_position(&files, id, ByteIndex(range.end as u32))
                                    .unwrap(),
                            ),
                            severity: Some(match severity {
                                Severity::Bug => lsp::DiagnosticSeverity::Error,
                                Severity::Error => lsp::DiagnosticSeverity::Error,
                                Severity::Warning => lsp::DiagnosticSeverity::Warning,
                                Severity::Note => lsp::DiagnosticSeverity::Information,
                                Severity::Help => lsp::DiagnosticSeverity::Hint,
                            }),
                            code: code.clone().map(lsp::NumberOrString::String),
                            source: Some(source.clone()),
                            message,
                            related_information: None,
                            tags: None,
                        })
                        .collect::<Vec<lsp::Diagnostic>>()
                },
            )
            .collect::<Vec<lsp::Diagnostic>>()
    }
}
