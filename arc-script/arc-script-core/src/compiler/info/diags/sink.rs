pub use std::io::Write;
pub use codespan_reporting::term::termcolor::StandardStream;
pub use codespan_reporting::term::termcolor::ColorChoice;
pub use codespan_reporting::term::termcolor::Buffer;
pub use codespan_reporting::term::termcolor::WriteColor;

use crate::compiler::shared::New;

#[derive(Default, New)]
pub struct Sink;

impl WriteColor for Sink {
    fn supports_color(&self) -> bool {
        true
    }

    fn set_color(
        &mut self,
        _: &codespan_reporting::term::termcolor::ColorSpec,
    ) -> std::io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Write for Sink {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

