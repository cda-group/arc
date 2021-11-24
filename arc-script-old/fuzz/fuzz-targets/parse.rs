#![no_main]

use arc_script_compiler::prelude::diags::sink::Sink;
use arc_script_compiler::prelude::modes::{Input, Mode, Output};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|source: String| {
    let sink = Sink::new();
    let mode = Mode {
        input: Input::Code(source),
        output: Output::AST,
        ..Default::default()
    };
    let _ = arc_script_compiler::compile(mode, sink);
});
