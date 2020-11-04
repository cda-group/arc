use {
    crate::prelude::*,
    anyhow::Result,
    linefeed::{Completer, Completion, Interface, Prompter, ReadResult, Terminal},
    std::sync::Arc,
};

pub fn start(opt: &Opt) -> Result<()> {
    let reader = Interface::new("arc-script")?;
    reader.set_prompt("λ ")?;
    reader.set_completer(Arc::new(Repl));
    while let ReadResult::Input(input) = reader.read_line()? {
        compiler::diagnose(&input, opt);
        reader.set_buffer(&input)?;
    }
    Ok(())
}

#[rustfmt::skip]
static COMMANDS: &[(&str, &str)] = &[
    ("set",              "Assign a value to a variable"),
    ("get",              "Print the value of a variable"),
    ("help",             "You're looking at it"),
    ("list-commands",    "List command names"),
    ("list-variables",   "List variables"),
    ("history",          "Print history"),
    ("save-history",     "Write history to file"),
    ("quit",             "Quit the REPL"),
];

struct Repl;

impl<T: Terminal> Completer<T> for Repl {
    fn complete(
        &self,
        word: &str,
        prompter: &Prompter<T>,
        start: usize,
        _end: usize,
    ) -> Option<Vec<Completion>> {
        let mut words = prompter.buffer()[..start].split_whitespace();

        match words.next() {
            // Complete command name
            None => {
                let mut res = Vec::new();
                for (cmd, _) in COMMANDS {
                    if cmd.starts_with(word) {
                        res.push(Completion::simple(cmd.to_string()));
                    }
                }
                Some(res)
            }
            // Complete command parameters
            Some("get") | Some("set") => {
                if words.count() == 0 {
                    let mut res = Vec::new();
                    for (name, _) in prompter.variables() {
                        if name.starts_with(word) {
                            res.push(Completion::simple(name.to_owned()));
                        }
                    }
                    Some(res)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
