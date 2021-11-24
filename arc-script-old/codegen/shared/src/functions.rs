//! Functions must be cloneable

use dyn_clone::DynClone;

pub trait ValueFn<Args>: FnOnce<Args> + Send + DynClone + 'static {}
impl<Args, F> ValueFn<Args> for F where F: FnOnce<Args> + Send + DynClone + 'static {}

dyn_clone::clone_trait_object!(<I, O> ValueFn<I, Output=O>);

/// Construct a new function value.
#[macro_export]
macro_rules! fun_val {
    { $arg:expr } => { Box::new($arg) }
}

/// Construct a new function type.
#[macro_export]
macro_rules! fun_type {
    { ($($inputs:ty),+) -> $output:ty } => { Box<dyn ValueFn($($inputs),+) -> $output> }
}
