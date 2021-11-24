use arc_script_compiler::prelude::diags::{ColorChoice, StandardStream};
use arc_script_compiler::prelude::modes::{Input, Mode};

use rustyline::error::ReadlineError;
use rustyline::Editor;

use anyhow::Result;
use std::io::Write;

pub fn start(mode: Mode) -> Result<()> {
    let mut repl = Editor::<()>::new();
    let _ = repl.load_history("history.txt");
    let f = StandardStream::stdout(ColorChoice::Always);
    let mut f = f.lock();
    loop {
        match repl.readline("λ ") {
            Ok(source) => {
                repl.add_history_entry(source.as_str());
                let run = Mode {
                    input: Input::Code(source),
                    ..mode.clone()
                };
                arc_script_compiler::compile(run, &mut f)?;
            }
            Err(ReadlineError::Interrupted) => {
                write!(f, "CTRL-C")?;
                break;
            }
            Err(ReadlineError::Eof) => {
                write!(f, "CTRL-D")?;
                break;
            }
            Err(err) => {
                write!(f, "Error: {:?}", err)?;
                break;
            }
        }
    }
    repl.save_history("history.txt")?;
    Ok(())
}
