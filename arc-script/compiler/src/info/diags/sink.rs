use arc_script_compiler_shared::New;

pub use codespan_reporting::term::termcolor::Buffer;
pub use codespan_reporting::term::termcolor::ColorChoice;
pub use codespan_reporting::term::termcolor::StandardStream;
pub use codespan_reporting::term::termcolor::WriteColor;

use std::io::Result;

use codespan_reporting::term::termcolor::ColorSpec;

/// Trait which wraps `Write` and `WriteColor`.
pub trait Writer: std::io::Write + WriteColor {}

impl<T> Writer for T where T: std::io::Write + WriteColor {}

/// A writer which will move data into the void.
/// Just like [`std::io::Sink`], but supports the `termcolor` library.
#[derive(Debug, Default, New)]
pub struct Sink;

impl WriteColor for Sink {
    fn supports_color(&self) -> bool {
        true
    }

    fn set_color(&mut self, _: &ColorSpec) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

impl std::io::Write for Sink {
    fn write(&mut self, _: &[u8]) -> Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
