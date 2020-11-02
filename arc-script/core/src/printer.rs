use crate::prelude::*;

#[derive(Copy, Clone)]
pub struct Printer<'a> {
    pub info: &'a Info<'a>,
    pub tabs: u32,
    pub verbose: bool,
}

const TAB: &str = "    ";

impl Printer<'_> {
    pub fn indent(&self) -> String {
        format!("\n{}", (0..self.tabs).map(|_| TAB).collect::<String>())
    }

    pub fn tab(&self) -> Printer {
        Printer {
            info: self.info,
            tabs: self.tabs + 1,
            verbose: self.verbose,
        }
    }

    pub fn untab(&self) -> Printer {
        Printer {
            info: self.info,
            tabs: self.tabs - 1,
            verbose: self.verbose,
        }
    }
    pub fn lookup(&self, tv: TypeVar) -> Type {
        self.info.typer.borrow_mut().lookup(tv)
    }
}

impl<'i> From<&'i Info<'_>> for Printer<'i> {
    fn from(info: &'i Info) -> Self {
        Self {
            info,
            tabs: 0,
            verbose: false,
        }
    }
}

impl<'i> From<&'i mut Info<'_>> for Printer<'i> {
    fn from(info: &'i mut Info) -> Self {
        Self {
            info,
            tabs: 0,
            verbose: false,
        }
    }
}
