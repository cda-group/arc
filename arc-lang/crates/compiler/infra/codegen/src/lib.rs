use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::io;
use std::io::LineWriter;
use std::io::Write;
use std::path::PathBuf;

use colored::Color;
pub use colored::Color::*;

use colored::Colorize;
use colors::*;
use config::Show;

pub struct Context<W: Write> {
    writer: W,
    indent: usize,
    depth: usize,
    pub opt: Show,
}

impl Context<Vec<u8>> {
    pub fn string() -> Self {
        Self::new(Vec::new())
    }

    pub fn finish(&mut self) -> String {
        String::from_utf8(std::mem::take(&mut self.writer)).unwrap()
    }
}

impl Context<LineWriter<std::fs::File>> {
    pub fn file(path: PathBuf) -> Self {
        let file = std::fs::File::create(path).unwrap();
        Self::new(LineWriter::new(file))
    }
}

impl Context<LineWriter<io::Stderr>> {
    pub fn stderr() -> Self {
        Self::new(LineWriter::new(io::stderr()))
    }
}

impl<W: Write> Context<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            indent: 0,
            depth: 0,
            opt: Show::default(),
        }
    }

    pub fn with_opt(&mut self, opt: Show) -> &mut Self {
        self.opt = opt;
        self
    }

    pub fn colors(&mut self, colors: bool) -> &mut Self {
        self.opt.colors = colors;
        self
    }

    pub fn write<T>(
        &mut self,
        x: T,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<&mut Self> {
        f(self, x)?;
        self.flush();
        Ok(self)
    }

    pub fn writeln<T>(
        &mut self,
        x: T,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<&mut Self> {
        f(self, x)?;
        self.writer.write(b"\n")?;
        self.flush();
        Ok(self)
    }

    pub fn typed(&mut self) -> &mut Self {
        self.opt.types = true;
        self
    }

    pub fn debug(&mut self) -> &mut Self {
        self.opt.prelude = true;
        self
    }

    pub fn flush(&mut self) {
        self.writer.flush().unwrap();
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn indented(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.indent();
        f(self)?;
        self.dedent();
        Ok(())
    }

    pub fn indented_if_toplevel(
        &mut self,
        f: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.depth == 0 {
            self.indent();
            self.newline()?;
            self.dedent();
        }
        f(self)
    }

    pub fn comma(&mut self) -> io::Result<()> {
        self.lit(", ")
    }

    pub fn colon(&mut self) -> io::Result<()> {
        self.lit(": ")
    }

    pub fn space(&mut self) -> io::Result<()> {
        self.lit(" ")
    }

    pub fn keyword(&mut self, s: &str) -> io::Result<()> {
        self.bold_colored(s, KEYWORD_COLOR)
    }

    pub fn mac(&mut self, s: &str) -> io::Result<()> {
        self.bold_colored(s, MACRO_COLOR)
    }

    pub fn dbg(&mut self, s: impl Debug) -> io::Result<()> {
        self.fmt(format_args!("{:?}", s))
    }

    pub fn lit(&mut self, s: impl Display) -> io::Result<()> {
        self.fmt(format_args!("{}", s))
    }

    pub fn def(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, DEF_COLOR)
    }

    pub fn var(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, VAR_COLOR)
    }

    pub fn val(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, VAL_COLOR)
    }

    pub fn builtin(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, BUILTIN_COLOR)
    }

    pub fn numeric(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, NUMERIC_COLOR)
    }

    pub fn comment(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, COMMENT_COLOR)
    }

    pub fn ty(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, TYPE_COLOR)
    }

    pub fn text(&mut self, s: &str) -> io::Result<()> {
        self.colored(s, STRING_COLOR)
    }

    pub fn colored(&mut self, s: &str, c: Color) -> io::Result<()> {
        if self.opt.colors {
            self.fmt(format_args!("{}", s.color(c)))
        } else {
            self.lit(s)
        }
    }

    pub fn bold_colored(&mut self, s: &str, c: Color) -> io::Result<()> {
        if self.opt.colors {
            self.fmt(format_args!("{}", s.color(c).bold()))
        } else {
            self.lit(s)
        }
    }

    pub fn display(&mut self, x: impl fmt::Display) -> io::Result<()> {
        self.fmt(format_args!("{}", x))
    }

    pub fn fmt(&mut self, args: fmt::Arguments) -> io::Result<()> {
        self.writer.write_fmt(args)
    }

    pub fn newline(&mut self) -> io::Result<()> {
        self.lit("\n")?;
        self.tab()?;
        Ok(())
    }

    pub fn tab(&mut self) -> io::Result<()> {
        for _ in 0..self.indent {
            self.lit("    ")?;
        }
        Ok(())
    }

    pub fn newline_if_toplevel(&mut self) -> io::Result<()> {
        if self.depth == 0 {
            self.newline()?;
        }
        Ok(())
    }

    pub fn newline_seq<'a, T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        for x in v {
            self.newline()?;
            f(self, x)?;
        }
        Ok(())
    }

    pub fn indented_seq<'a, T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        self.indent();
        self.newline_seq(v, f)?;
        self.dedent();
        self.newline()
    }

    pub fn indented_comma_seq<'a, T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        self.indent();
        self.newline()?;
        let mut iter = v.into_iter();
        if let Some(x) = iter.next() {
            f(self, x)?;
        }
        for x in iter {
            self.lit(",")?;
            self.newline()?;
            f(self, x)?;
        }
        self.dedent();
        self.newline()
    }

    pub fn then<T>(
        &mut self,
        x: &Option<T>,
        f: impl FnOnce(&mut Self, &T) -> io::Result<()>,
    ) -> io::Result<()> {
        if let Some(x) = x {
            f(self, x)?;
        }
        Ok(())
    }

    pub fn then_or<T>(
        &mut self,
        x: &Option<T>,
        f0: impl FnOnce(&mut Self, &T) -> io::Result<()>,
        f1: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        if let Some(x) = x {
            f0(self, x)?;
        } else {
            f1(self)?;
        }
        Ok(())
    }

    pub fn each_newline<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut iter = v.into_iter();
        if let Some(x) = iter.next() {
            f(self, x)?;
            while let Some(x) = iter.next() {
                self.newline()?;
                f(self, x)?;
            }
        }
        Ok(())
    }

    pub fn each<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        for i in v {
            f(self, i)?;
        }
        Ok(())
    }

    pub fn sep<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        sep: &str,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut iter = v.into_iter();
        if let Some(x) = iter.next() {
            f(self, x)?;
            while let Some(x) = iter.next() {
                self.lit(sep)?;
                self.lit(" ")?;
                f(self, x)?;
            }
        }
        Ok(())
    }

    pub fn sep_trailing<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        sep: &str,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut iter = v.into_iter();
        if let Some(x) = iter.next() {
            f(self, x)?;
            self.lit(sep)?;
            if let Some(x) = iter.next() {
                self.lit(" ")?;
                f(self, x)?;
                while let Some(x) = iter.next() {
                    self.lit(sep)?;
                    self.lit(" ")?;
                    f(self, x)?;
                }
            }
        }
        Ok(())
    }

    pub fn seq<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        self.sep(v, ",", f)
    }

    pub fn seq_trailing<T>(
        &mut self,
        v: impl IntoIterator<Item = T>,
        f: impl Fn(&mut Self, T) -> io::Result<()>,
    ) -> io::Result<()> {
        self.sep_trailing(v, ",", f)
    }

    pub fn delim(
        &mut self,
        l: &str,
        r: &str,
        f: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        self.lit(l)?;
        f(self)?;
        self.lit(r)?;
        Ok(())
    }

    pub fn block(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.lit("{")?;
        self.indent();
        f(self)?;
        self.dedent();
        self.newline()?;
        self.lit("}")?;
        self.depth -= 1;
        Ok(())
    }

    pub fn brace(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.delim("{", "}", f)?;
        self.depth -= 1;
        Ok(())
    }

    pub fn quote(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.delim("\"", "\"", f)?;
        self.depth -= 1;
        Ok(())
    }

    pub fn paren(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.delim("(", ")", f)?;
        self.depth -= 1;
        Ok(())
    }

    pub fn brack(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.delim("[", "]", f)?;
        self.depth -= 1;
        Ok(())
    }

    pub fn angle(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.depth += 1;
        self.delim("<", ">", f)?;
        self.depth -= 1;
        Ok(())
    }

    pub fn annot(&mut self, f: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()> {
        self.lit(": ")?;
        f(self)
    }
}
