mod utils;

use arc_script_compiler::prelude::diags::Buffer;
use arc_script_compiler::prelude::modes::{Input, Mode, Output};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use xterm_js_rs::addons::fit::FitAddon;
use xterm_js_rs::{OnKeyEvent, Terminal, TerminalOptions, Theme};

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Basic function for compiling and generating source code
fn compile(source: &mut String) -> String {
    let source = std::mem::take(source);
    let mut sink = Buffer::no_color();
    let mode = Mode {
        input: Input::Code(source),
        output: Output::MLIR,
        ..Default::default()
    };
    arc_script_compiler::compile(mode, &mut sink)
        .map(|_| {
            std::str::from_utf8(sink.as_slice())
                .map(ToString::to_string)
                .unwrap_or_else(|e| format!("Internal Error: {}", e))
        })
        .unwrap_or_else(|e| format!("Internal Error: {}", e))
}

const PROMPT: &str = "$ ";

fn prompt(term: &Terminal) {
    term.writeln("");
    term.write(PROMPT);
}

// Keyboard keys
// https://notes.burke.libbey.me/ansi-escape-codes/
const KEY_ENTER: u32 = 13;
const KEY_BACKSPACE: u32 = 8;
const KEY_LEFT_ARROW: u32 = 37;
const KEY_RIGHT_ARROW: u32 = 39;
const KEY_C: u32 = 67;
const KEY_L: u32 = 76;

const CURSOR_LEFT: &str = "\x1b[D";
const CURSOR_RIGHT: &str = "\x1b[C";

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    utils::_set_panic_hook();

    let terminal: Terminal = Terminal::new(
        TerminalOptions::new()
            .with_rows(50)
            .with_cursor_blink(true)
            .with_cursor_width(10)
            .with_font_size(20)
            .with_draw_bold_text_in_bright_colors(true)
            .with_right_click_selects_word(true)
            .with_theme(
                Theme::new()
                    .with_foreground("#98FB98")
                    .with_background("#000000"),
            ),
    );

    let elem = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("terminal")
        .unwrap();

    terminal.writeln("Supported keys in this example: <Printable-Characters> <Enter> <Backspace> <Left-Arrow> <Right-Arrow> <Ctrl-C> <Ctrl-L>");
    terminal.open(elem.dyn_into()?);
    prompt(&terminal);

    let mut line = String::new();
    let mut cursor_col = 0;

    let term: Terminal = terminal.clone().dyn_into()?;

    let callback = Closure::wrap(Box::new(move |e: OnKeyEvent| {
        let event = e.dom_event();
        match event.key_code() {
            KEY_ENTER => {
                if !line.is_empty() {
                    let code = compile(&mut line);
                    term.writeln(&code);
                    term.writeln("");
                    line.clear();
                    cursor_col = 0;
                }
                prompt(&term);
            }
            KEY_BACKSPACE => {
                if cursor_col > 0 {
                    term.write("\u{0008} \u{0008}");
                    line.pop();
                    cursor_col -= 1;
                }
            }
            KEY_LEFT_ARROW => {
                if cursor_col > 0 {
                    term.write(CURSOR_LEFT);
                    cursor_col -= 1;
                }
            }
            KEY_RIGHT_ARROW => {
                if cursor_col < line.len() {
                    term.write(CURSOR_RIGHT);
                    cursor_col += 1;
                }
            }
            KEY_L if event.ctrl_key() => term.clear(),
            KEY_C if event.ctrl_key() => {
                prompt(&term);
                line.clear();
                cursor_col = 0;
            }
            _ => {
                if !event.alt_key() && !event.alt_key() && !event.ctrl_key() && !event.meta_key() {
                    term.write(&event.key());
                    line.push_str(&e.key());
                    cursor_col += 1;
                }
            }
        }
    }) as Box<dyn FnMut(_)>);

    terminal.on_key(callback.as_ref().unchecked_ref());

    callback.forget();

    let addon = FitAddon::new();
    terminal.load_addon(addon.clone().dyn_into::<FitAddon>()?.into());
    addon.fit();
    terminal.focus();

    Ok(())
}
