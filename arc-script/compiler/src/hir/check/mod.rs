use crate::hir::HIR;
use crate::info::Info;

// mod ownership;

impl HIR {
    pub(crate) fn check(&self, info: &mut Info) {
//         self.check_ownership(info);
    }
}

// TODO:
// * Check that Rust-functions are not called at staging-time (since they may not yet be compiled)
//   * Option 1: Treat external functions as a different type
//      * Disallow them to be called at staging time
//      * Allow them to be passed by-reference
//      * Regular functions are upcast to external (non-callable) functions
//   * Option 2: Handle it at runtime
//   * Option 3: Force external functions to only be used inside operators
// * Check that all exported public Arc functions only take streams as input/output
