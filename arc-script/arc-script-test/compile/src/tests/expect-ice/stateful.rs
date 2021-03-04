// compile-flags: --error-format=human
#[arc_script::compile("stateful.arc")]
mod script {}

fn main() {}

/// Extern User-Defined-Implementation
impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
    script::Stateful<'i, 'source, 'timer, 'channel, B, C>
{
    /// Extern User-Defined-Method
    fn update(&mut self) -> i32 {
        self.state.value += 1;
        self.state.value
    }
}
