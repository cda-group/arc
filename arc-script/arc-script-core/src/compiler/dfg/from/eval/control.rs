use crate::compiler::dfg::from::eval::value::Value;
use crate::compiler::info::files::Loc;

/// Trait for panicking the interpreter through unwinding the stackframes.
pub(crate) trait Unwind<T> {
    /// Unwinds if the value contained inside is a `None` or `Err`.
    fn or_unwind(self, loc: Option<Loc>) -> std::result::Result<T, ControlKind>;
}

impl<T> Unwind<T> for Option<T> {
    fn or_unwind(self, loc: Option<Loc>) -> std::result::Result<T, ControlKind> {
        self.map_or(Control(ControlKind::Panic(loc)), Ok)
    }
}

/// Result of evaluating a script.
pub(crate) type EvalResult = std::result::Result<Value, ControlKind>;

/// Control-flow is managed through Rust-exceptions.
pub(crate) use std::result::Result::Err as Control;

/// Different kinds of control-flow instructions (exceptions).
pub(crate) enum ControlKind {
    /// Returns value by popping the current stack-frame.
    Return(Value),
    /// Breaks value out of the current loop.
    Break(Value),
    /// Panics and unwinds all stack-frames.
    Panic(Option<Loc>),
}
