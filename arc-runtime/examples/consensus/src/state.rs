use arc_runtime::prelude::*;

#[derive(Clone, Debug, New)]
pub struct State {
    pub key: String,
    pub value: u64,
}

impl IntoSendable for State {
    type T = State;
    fn into_sendable(self) -> Self::T {
        todo!()
    }
}
