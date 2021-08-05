// compile-flags: --error-format=human

use arcon::prelude::Backend;
use arcon::prelude::ComponentDefinition;

mod script {
    arc_script::include!("src/tests/expect_mlir_fail_todo/extern_fun.rs");
    use super::increment;
}

/// Extern User-Defined-Function
pub(crate) fn increment(x: i32) -> i32 {
    x + 1
}

/// Extern User-Defined-Implementation
impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
    script::Adder<'i, 'source, 'timer, 'channel, B, C>
{
    /// Extern User-Defined-Method
    fn addition(&mut self, x: i32, y: i32) -> i32 {
        x + y
    }
}

fn main() {}
