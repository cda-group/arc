use kompact::prelude::*;

#[derive(ComponentDefinition, Actor)]
pub struct Client {
    pub ctx: ComponentContext<Self>,
}

impl Client {
    pub(crate) fn new() -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
        }
    }
}

impl ComponentLifecycle for Client {}
