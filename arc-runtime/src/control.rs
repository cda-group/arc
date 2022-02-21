use std::ops::ControlFlow;
use std::ops::FromResidual;
use std::ops::Try;

#[derive(Debug)]
pub enum Control<T> {
    Continue(T),
    Finished,
}

impl<T> FromResidual<()> for Control<T> {
    fn from_residual(residual: ()) -> Self {
        Control::Finished
    }
}

impl<T> Try for Control<T> {
    type Output = T;
    type Residual = ();
    fn from_output(output: T) -> Self {
        Control::Continue(output)
    }
    fn branch(self) -> ControlFlow<(), T> {
        match self {
            Control::Continue(output) => ControlFlow::Continue(output),
            Control::Finished => ControlFlow::Break(()),
        }
    }
}
