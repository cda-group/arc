use arc_script_core::compiler::info::diags;
use arc_script_core::compiler::info::diags::to_codespan::ToCodespan;
use arc_script_core::compiler::info::diags::to_codespan::{Codespan, Report};
use arc_script_core::compiler::info::files;
use arc_script_core::compiler::info::Info;

use codespan_lsp::byte_index_to_position;
use codespan_reporting::diagnostic::{Diagnostic, Label, Severity};
use tower_lsp::lsp_types as lsp;

#[rustfmt::skip]
pub fn to_lsp(report: Report) -> Vec<lsp::Diagnostic> {
    let (diags, info) = report.into();
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
                    .map(|Label { file_id, range, message, .. }| lsp::Diagnostic {
                        range: lsp::Range::new(
                            byte_index_to_position(&info.files.store, file_id, range.start).unwrap(),
                            byte_index_to_position(&info.files.store, file_id, range.end).unwrap(),
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
