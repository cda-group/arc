use arc_script_core_shared::New;

pub use codespan_reporting::term::termcolor::Buffer;
pub use codespan_reporting::term::termcolor::ColorChoice;
pub use codespan_reporting::term::termcolor::StandardStream;
pub use codespan_reporting::term::termcolor::WriteColor;

use std::io::Result;
pub use std::io::Write;

use codespan_reporting::term::termcolor::ColorSpec;

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

impl Write for Sink {
    fn write(&mut self, _: &[u8]) -> Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
