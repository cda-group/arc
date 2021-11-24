use arc_script_compiler::info::diags;
use arc_script_compiler::info::diags::to_codespan::Codespan;
use arc_script_compiler::info::diags::to_codespan::Report;
use arc_script_compiler::info::diags::to_codespan::ToCodespan;
use arc_script_compiler::info::files;
use arc_script_compiler::info::Info;

use codespan_lsp::byte_index_to_position;
use codespan_reporting::diagnostic::Diagnostic;
use codespan_reporting::diagnostic::Label;
use codespan_reporting::diagnostic::Severity;
use lspower::lsp;

#[rustfmt::skip]
pub fn to_lsp(report: Report) -> Vec<lsp::Diagnostic> {
    let (diags, info) = report.into();
    diags
        .into_iter()
        .flat_map(
            |Diagnostic {
                 severity,
                 code,
                 message,
                 labels,
                 ..
             }| {
                labels
                    .into_iter()
                    .map(|Label { file_id, range, message, .. }| lsp::Diagnostic {
                        range: lsp::Range::new(
                            byte_index_to_position(&info.files.store, file_id, range.start).unwrap(),
                            byte_index_to_position(&info.files.store, file_id, range.end).unwrap(),
                        ),
                        severity: match severity {
                            Severity::Bug => lsp::DiagnosticSeverity::Error,
                            Severity::Error => lsp::DiagnosticSeverity::Error,
                            Severity::Warning => lsp::DiagnosticSeverity::Warning,
                            Severity::Note => lsp::DiagnosticSeverity::Information,
                            Severity::Help => lsp::DiagnosticSeverity::Hint,
                        }.into(),
                        code: code.clone().map(lsp::NumberOrString::String),
                        source: message.clone().into(),
                        message,
                        related_information: None,
                        tags: None,
                        code_description: None,
                        data: None,
                    })
                    .collect::<Vec<lsp::Diagnostic>>()
            },
        )
        .collect::<Vec<lsp::Diagnostic>>()
}
