#[derive(prost::Message, Copy, Clone, Eq, PartialEq)]
pub struct Unit {}

impl Unit {
    pub fn new() -> Self {
        Self {}
    }
}

pub use half::bf16;
pub use half::f16;
