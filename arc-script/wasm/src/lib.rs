mod utils;

use arc_script::pretty::Pretty;
use arc_script::opt::*;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn compile(source: &str) -> String {
    let opt = Opt {
        debug: false,
        mlir: true,
        verbose: false,
        subcmd: SubCmd::Lib,
    };
    let (script, reporter) = arc_script::compile(source, &opt);
    if reporter.diags.is_empty() {
        script.body.mlir()
    } else {
        reporter.emit_as_str()
    }
}
