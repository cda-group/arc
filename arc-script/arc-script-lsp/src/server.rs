use crate::completer::*;
use crate::linter;

use arc_script_core::prelude::compiler;
use arc_script_core::prelude::diags::Sink;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use serde_json::{Number, Value};
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

#[derive(Debug)]
pub(crate) struct Backend {
    pub(crate) client: Client,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: None,
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::Full,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(true),
                    trigger_characters: Some(vec![
                        ".".to_string(),
                        r"\".to_string(),
                        ":".to_string(),
                    ]),
                    work_done_progress_options: WorkDoneProgressOptions {
                        work_done_progress: Some(true),
                    },
                }),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: None,
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                document_highlight_provider: Some(true),
                workspace_symbol_provider: Some(true),
                execute_command_provider: Some(ExecuteCommandOptions {
                    commands: vec!["dummy.do_something".to_string()],
                    work_done_progress_options: Default::default(),
                }),
                workspace: Some(WorkspaceCapability {
                    workspace_folders: Some(WorkspaceFolderCapability {
                        supported: Some(true),
                        change_notifications: Some(
                            WorkspaceFolderCapabilityChangeNotifications::Bool(true),
                        ),
                    }),
                }),
                ..ServerCapabilities::default()
            },
        })
    }

    async fn completion(&self, p: CompletionParams) -> Result<Option<CompletionResponse>> {
        //        let start = p.text_document_position.position;
        let items = LATEX
            .iter()
            .take(20)
            .map(|(key, value)| {
                //                 let len = value.len() as u64;
                //                 let end = Position { line: start.line, character: start.character+len };
                CompletionItem {
                    label: key.to_string(),
                    detail: Some(value.to_string()),
                    //                 text_edit: Some(TextEdit{range: Range{start, end}, new_text:value.to_string()}),
                    //                 insert_text: Some(value.to_string()),
                    data: Some(Value::Number(Number::from(
                        p.text_document_position.position.character,
                    ))),
                    ..Default::default()
                }
            })
            .collect();
        Ok(Some(CompletionResponse::List(CompletionList {
            is_incomplete: true,
            items,
        })))
    }

    async fn completion_resolve(&self, mut p: CompletionItem) -> Result<CompletionItem> {
        // p.insert_text = p.detail.clone();

        if let (Some(Value::Number(n)), Some(text)) = (p.data.clone(), p.detail.clone()) {
            p.insert_text_format = Some(InsertTextFormat::PlainText);
            let n = n.as_u64().unwrap();
            let a = Position::new(1, n - 1);
            let b = Position::new(1, n + text.chars().count() as u64 - 1);
            p.text_edit = Some(CompletionTextEdit::Edit(TextEdit {
                range: Range { start: a, end: b },
                new_text: text,
            }));
            Ok(p)
        } else {
            Ok(p)
        }
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::Info, "Arc-Script LSP - ONLINE!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_change_workspace_folders(&self, _: DidChangeWorkspaceFoldersParams) {
        self.client
            .log_message(MessageType::Info, "workspace folders changed!")
            .await;
    }

    async fn did_change_configuration(&self, _: DidChangeConfigurationParams) {
        self.client
            .log_message(MessageType::Info, "configuration changed!")
            .await;
    }

    async fn did_change_watched_files(&self, _: DidChangeWatchedFilesParams) {
        self.client
            .log_message(MessageType::Info, "watched files have changed!")
            .await;
    }

    async fn execute_command(&self, _: ExecuteCommandParams) -> Result<Option<Value>> {
        self.client
            .log_message(MessageType::Info, "command executed!")
            .await;

        match self.client.apply_edit(WorkspaceEdit::default()).await {
            Ok(res) if res.applied => {
                self.client
                    .log_message(MessageType::Info, "edit applied")
                    .await
            }
            Ok(_) => {
                self.client
                    .log_message(MessageType::Info, "edit not applied")
                    .await
            }
            Err(_) => self.client.log_message(MessageType::Error, "error").await,
        };

        Ok(None)
    }

    async fn did_open(&self, _: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::Info, "file opened!")
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::Info, "file changed!")
            .await;
        let uri = params.text_document.uri;
        let source = params.content_changes.into_iter().next().unwrap().text;
        report(&self.client, uri, source).await;
    }

    async fn did_save(&self, _: DidSaveTextDocumentParams) {
        self.client
            .log_message(MessageType::Info, "file saved!")
            .await;
    }

    async fn did_close(&self, _: DidCloseTextDocumentParams) {
        self.client
            .log_message(MessageType::Info, "file closed!")
            .await;
    }
}

async fn report(client: &Client, uri: Url, source: String) {
    let mode = Mode {
        input: Input::Code(source),
        output: Output::Silent,
        ..Default::default()
    };
    let sink = Sink::default();
    if let Ok(report) = compiler::compile(mode, sink) {
        let diagnostics = linter::to_lsp(report);
        client
            .publish_diagnostics(uri.clone(), diagnostics, None)
            .await
    }
}

#[tokio::main]
pub async fn start(mode: Mode) {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, messages) = LspService::new(|client| Backend { client });
    Server::new(stdin, stdout)
        .interleave(messages)
        .serve(service)
        .await;
}
