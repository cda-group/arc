//! Functions must be cloneable

use dyn_clone::DynClone;

pub trait ArcornFn<Args>: FnOnce<Args> + Send + DynClone + 'static {}
impl<Args, F> ArcornFn<Args> for F where F: FnOnce<Args> + Send + DynClone + 'static {}

dyn_clone::clone_trait_object!(<I, O> ArcornFn<I, Output=O>);
