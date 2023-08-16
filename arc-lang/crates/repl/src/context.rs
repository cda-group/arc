use anyhow::Result;
use colored::Color;
use im_rc::Vector;
use rustyline::config::Configurer;
use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::history::MemHistory;
use rustyline::Cmd;
use rustyline::CompletionType;
use rustyline::EditMode;
use rustyline::Editor;
use rustyline::Helper;
use rustyline::KeyCode;
use rustyline::KeyEvent;
use rustyline::Modifiers;

use compiler::Compiler;

use super::helper::Repl;

#[derive(Debug)]
pub(crate) struct Context {
    pub(crate) count: usize,
    pub(crate) editor: Editor<crate::helper::Repl, FileHistory>,
    pub(crate) compiler: Compiler,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.save_history();
    }
}

impl Context {
    pub(crate) fn new(compiler: Compiler) -> Result<Self> {
        let mut this = Self {
            count: 0,
            editor: Editor::new()?,
            compiler,
        };
        this.editor.set_helper(Some(Repl::default()));
        this.config()?;
        this.color(Color::Green);
        if !this.compiler.config.history.exists() {
            std::fs::create_dir_all(this.compiler.config.history.parent().unwrap())?;
            std::fs::File::create(&this.compiler.config.history)?;
        }
        this.load_history();
        Ok(this)
    }

    pub(crate) fn save_history(&mut self) {
        self.editor.save_history(&self.compiler.config.history).ok();
    }

    pub(crate) fn load_history(&mut self) {
        self.editor.load_history(&self.compiler.config.history).ok();
    }

    pub(crate) fn color(&mut self, color: Color) {
        self.editor.helper_mut().unwrap().prompt_color = color;
    }

    pub(crate) fn readline_initial(
        &mut self,
        s: &str,
    ) -> std::result::Result<String, ReadlineError> {
        self.count += 1;
        self.editor.readline_with_initial(">> ", (s, ""))
    }

    pub(crate) fn readline(&mut self) -> std::result::Result<String, ReadlineError> {
        self.count += 1;
        self.editor.readline(">> ")
    }

    pub(crate) fn config(&mut self) -> Result<()> {
        self.editor.set_history_ignore_dups(true)?;
        self.editor.set_edit_mode(EditMode::Vi);
        self.editor.set_completion_type(CompletionType::List);
        self.editor
            .bind_sequence(KeyEvent::ctrl('j'), Cmd::NextHistory);
        self.editor
            .bind_sequence(KeyEvent::ctrl('k'), Cmd::PreviousHistory);
        self.editor
            .bind_sequence(KeyEvent::ctrl('l'), Cmd::ClearScreen);
        self.editor
            .bind_sequence(KeyEvent::ctrl('c'), Cmd::Interrupt);
        self.editor.bind_sequence(KeyEvent::ctrl('v'), Cmd::YankPop);
        self.editor.bind_sequence(
            KeyEvent::ctrl('M'),
            Cmd::AcceptOrInsertLine {
                accept_in_the_middle: false,
            },
        );
        self.editor.bind_sequence(
            KeyEvent(KeyCode::Enter, Modifiers::CTRL),
            Cmd::HistorySearchForward,
        );
        Ok(())
    }
}
