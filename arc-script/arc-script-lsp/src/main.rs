use arc_script_core::prelude::modes::{Input, Mode, Output};
use arc_script_lsp::runtime;

fn main() {
    let mode = Mode {
        input: Input::Empty,
        output: Output::Silent,
        ..Default::default()
    };
    runtime::start(mode).unwrap()
}
