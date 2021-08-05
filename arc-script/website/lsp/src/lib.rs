mod utils;

use arc_script_compiler::prelude::modes::{Input, Mode, Output};
use arc_script_lsp::server::Backend;

use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// #[wasm_bindgen]
// pub fn server(
//     input: node_sys::stream::Readable,
//     output: node_sys::stream::Writable,
// ) -> js_sys::Promise {
//     use js_sys_futures::{future_to_promise, JsAsyncRead};
//     use lspower::{LspService, Server};
//     use node_sys::JsAsyncWrite;
//
//     future_to_promise(async {
//         console_error_panic_hook::set_once();
//
//         let mode = Mode {
//             input: Input::Empty,
//             output: Output::Silent,
//             ..Default::default()
//         };
//
//         let (service, messages) = LspService::new(|client| Backend::new(client, mode));
//         let stdin = JsAsyncRead::new(input.into()).unwrap();
//         let stdout = JsAsyncWrite::new(output.into());
//         Server::new(stdin, stdout)
//             .interleave(messages)
//             .serve(service)
//             .await;
//
//         Ok(JsValue::undefined())
//     })
// }
