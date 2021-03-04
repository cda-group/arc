// compile-flags: --error-format=human

use arcon::prelude::ComponentDefinition;
use arcon::prelude::Backend;

#[arc_script::compile("extern-fun.arc")]
mod script {
    use super::increment;
}

/// Extern User-Defined-Function
pub(crate) fn increment(x: i32) -> i32 {
    x + 1
}

/// Extern User-Defined-Implementation
impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
    script::HandlerAdder<'i, 'source, 'timer, 'channel, script::TaskAdder, B, C>
{
    /// Extern User-Defined-Method
    fn addition(&mut self, x: i32, y: i32) -> i32 {
        x + y
    }
}

fn main() {}
