use {
    crate::error::{Diagnostic, Reporter, SimpleFile},
    codespan::ByteIndex,
    codespan_lsp::byte_index_to_position,
    codespan_reporting::diagnostic::{Label, Severity},
    tower_lsp::lsp_types as lsp,
};

impl<'i> Reporter<'i> {
    pub fn to_lsp(self) -> Vec<lsp::Diagnostic> { to_lsp(self.file, self.diags) }
}

fn to_lsp(file: SimpleFile, diags: Vec<Diagnostic>) -> Vec<lsp::Diagnostic> {
    let mut files = codespan::Files::new();
    let id = files.add("", file.source());
    diags
        .into_iter()
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
